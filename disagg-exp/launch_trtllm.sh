#!/usr/bin/env bash
# =============================================================================
# launch_trtllm.sh — TensorRT-LLM v1.2.1 PD-disaggregation 런처 (Qwen3-4B dense)
# -----------------------------------------------------------------------------
# 역할(ROLE): context | generation | proxy | all
#   context     : prefill 워커 NUM_CTX개 기동 (백그라운드)
#   generation  : decode  워커 NUM_GEN개 기동 (백그라운드)
#   proxy       : orchestrator(=trtllm-serve disaggregated) 기동 (포그라운드, :8000)
#   all         : 단일 노드 편의 — context+generation(bg) → health 대기 → proxy(fg)
#
# 모든 (TP,PP,xPyD,placement)은 환경변수로 파라미터화 — 하드코딩 config 분기 없음.
# 모든 플래그/키/포트는 v1.2.1 소스 대조 검증됨 (근거: disagg-exp/SETUP_LOG.md, CLAUDE.md).
#   - 모델은 positional 인자 (`--model`/`--served-model-name` 플래그 없음)
#   - --tp_size/--pp_size 존재(별칭), --dtype CLI 없음(→ extra YAML)
#   - orchestrator host/port는 CLI 불가 → disagg YAML(hostname/port)에서 읽음
#   - 워커 변인통제(dtype/attn_backend/kv_cache/cuda_graph)는 ctx/gen extra YAML 담당
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 공통 (변인통제: 한 실험 안에서 전 config 동일하게 둘 것) ----------
MODEL="${MODEL:-Qwen/Qwen3-4B}"          # positional model arg (HF id). 로컬경로 쓰면 sweep MODEL_NAME=basename 맞출 것
BACKEND="${BACKEND:-pytorch}"            # pytorch만 _torch/modeling_qwen3 경로 + dtype auto=bf16
LABEL="${LABEL:-T1}"                     # 로그/생성파일 라벨 (metadata·analyze 라벨과 일치시킬 것)
LOG_DIR="${EXP_LOG_DIR:-$SCRIPT_DIR/results}"
CTX_EXTRA="${CTX_EXTRA:-$SCRIPT_DIR/ctx_extra_llm_api_options.yaml}"   # context 워커 변인통제 YAML
GEN_EXTRA="${GEN_EXTRA:-$SCRIPT_DIR/gen_extra_llm_api_options.yaml}"   # generation 워커 변인통제 YAML
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"   # orchestrator -t (초). 대형모델 안전
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"             # orchestrator -r (초)
mkdir -p "$LOG_DIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
# Ethernet(EFA 없음) 노드용. NVLink 없는 g5/g6/g6e PCIe intra-node에도 안전.
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy,sm,self}"

# ---------- 로그레벨 / perf 메트릭 ----------
LOG_LEVEL="${LOG_LEVEL:-info}"            # 서버 로그레벨. 측정=info. 디버그=debug|verbose|trace. (워커 --log_level / orchestrator -l)
PERF_METRICS_MAX_REQUESTS="${PERF_METRICS_MAX_REQUESTS:-1000}"  # per-request perf(KV전송시간 등) 버퍼 크기. >=측정 N 권장. 0=비활성

# ===== DEBUG 토글 (기본 OFF) — ⚠️ 측정 런에선 OFF로 둘 것 (디버그 로그 = I/O 노이즈 → 변인 오염) =====
#   켜는 법: 아래 줄의 주석(#) 제거. 끄는 법: 다시 주석 처리. 각 줄 끝 = [이게 무슨 디버그인가].
#   (export된 env는 워커/orchestrator 자식 프로세스에 자동 상속됨.)
# export TLLM_LOG_LEVEL=debug                       # TRT-LLM 파이썬 상세 로그 (debug|verbose|trace)
# export UCX_LOG_LEVEL=debug                        # UCX(KV전송 전송계층) 상세 로그 — KV전송 디버깅용
# export TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1 # KV전송 overlap 비활성(타이밍 단순화) — KV전송 디버깅용
# ==============================================================================================

# ---------- 토폴로지 (config마다 바꾸는 독립변수) ----------
NUM_CTX="${NUM_CTX:-1}"                   # context 인스턴스 수 (xPyD의 P)
NUM_GEN="${NUM_GEN:-1}"                   # generation 인스턴스 수 (xPyD의 D; 1P3D면 3)
CTX_TP="${CTX_TP:-1}"; CTX_PP="${CTX_PP:-1}"   # prefill 병렬화 (비대칭축)
GEN_TP="${GEN_TP:-1}"; GEN_PP="${GEN_PP:-1}"   # decode  병렬화 (비대칭축)
CACHE_BACKEND="${CACHE_BACKEND:-UCX}"     # disagg YAML용(정보성). ⚠️ 실제 KV백엔드는 ctx/gen extra YAML이 결정
#                                           — 바꾸려면 ctx/gen_extra_llm_api_options.yaml의 cache_transceiver도 수정

# ---------- 포트 규약 (sweep.py가 :8000 조준 → proxy 절대 8000) ----------
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${PROXY_PORT:-8000}"
CTX_PORT_BASE="${CTX_PORT_BASE:-8001}"    # context 워커: 8001, 8002, ...
GEN_PORT_BASE="${GEN_PORT_BASE:-8011}"    # generation 워커: 8011, 8012, ...

# ---------- GPU 배치 (single-node 기본; inter-node는 각 노드서 *_GPU_BASE=0) ----------
ctx_ranks=$(( CTX_TP * CTX_PP ))          # 인스턴스당 GPU 수
gen_ranks=$(( GEN_TP * GEN_PP ))
CTX_GPU_BASE="${CTX_GPU_BASE:-0}"
GEN_GPU_BASE="${GEN_GPU_BASE:-$(( CTX_GPU_BASE + NUM_CTX * ctx_ranks ))}"

# ---------- disagg urls (intra=localhost 자동; inter=CTX_URLS/GEN_URLS 주입) ----------
# CTX_URLS/GEN_URLS: "host:port,host:port,..." (개수==NUM_CTX/NUM_GEN). 미설정 시 localhost+자동포트.
CTX_URLS="${CTX_URLS:-}"
GEN_URLS="${GEN_URLS:-}"

# ---------- server_role (독립워커 방식에선 보통 불필요 — 기본 0 권장) ----------
# README Basic Usage는 --server_role 없이 동작. mixed-precision/dynamic-scaling 예시만 사용.
# ⚠️ v1.2.1: metadata_server 없이 --server_role만 주면 enum 변환 경로(serve.py:548)를 안 타
#    문자열 그대로 전달됨 → 기본 0(생략)이 가장 안전. Phase 0서 필요 판명 시에만 1.
USE_SERVER_ROLE="${USE_SERVER_ROLE:-0}"

# =============================================================================
# 헬퍼
# =============================================================================
# GPU 슬라이스: base idx ranks → "g,g,g" (CUDA_VISIBLE_DEVICES용)
gpu_slice() {
  local base="$1" idx="$2" ranks="$3" start g out=""
  start=$(( base + idx * ranks ))
  for (( g=0; g<ranks; g++ )); do out+="$(( start + g )),"; done
  echo "${out%,}"
}

# urls 목록 생성: 명시값 우선, 없으면 localhost:port_base+idx
build_urls() {  # role(ctx|gen)
  local role="$1" n base explicit
  if [[ "$role" == "ctx" ]]; then n="$NUM_CTX"; base="$CTX_PORT_BASE"; explicit="$CTX_URLS"
  else n="$NUM_GEN"; base="$GEN_PORT_BASE"; explicit="$GEN_URLS"; fi
  if [[ -n "$explicit" ]]; then
    # 콤마 분리 + 토큰별 공백 트림 ("hostA:8001, hostB:8002" 같은 흔한 형식 방어)
    tr ',' '\n' <<< "$explicit" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
  else
    local i; for (( i=0; i<n; i++ )); do echo "localhost:$(( base + i ))"; done
  fi
}

# 워커 1개 기동 (백그라운드). role tp pp port gpus extra_yaml
launch_worker() {
  local role="$1" tp="$2" pp="$3" port="$4" gpus="$5" extra="$6"
  local role_flag=()
  if [[ "$USE_SERVER_ROLE" == "1" ]]; then
    [[ "$role" == "context" ]] && role_flag=(--server_role CONTEXT) || role_flag=(--server_role GENERATION)
  fi
  echo "[launch] $role tp=$tp pp=$pp gpus=[$gpus] port=$port extra=$(basename "$extra")" >&2
  CUDA_VISIBLE_DEVICES="$gpus" \
  trtllm-serve "$MODEL" \
    --backend "$BACKEND" \
    --log_level "$LOG_LEVEL" \
    --tp_size "$tp" \
    --pp_size "$pp" \
    --host 0.0.0.0 \
    --port "$port" \
    "${role_flag[@]+"${role_flag[@]}"}" \
    --extra_llm_api_options "$extra" \
    > "$LOG_DIR/trtllm_${LABEL}_${role}_p${port}_$(hostname).log" 2>&1 &
  echo "$!"   # PID
}

wait_health() {  # host port [timeout_s]
  local host="$1" port="$2" to="${3:-600}" t=0
  until curl -sf "http://${host}:${port}/health" >/dev/null 2>&1; do
    sleep 2; t=$(( t + 2 ))
    if (( t >= to )); then echo "[ERROR] health timeout ${host}:${port}" >&2; return 1; fi
  done
  echo "[ok] healthy ${host}:${port}"
}

# disagg orchestrator config YAML 생성 (런별로 LOG_DIR에). 이게 유일한 disagg config 생성 경로(정적 파일 없음).
write_disagg_yaml() {
  local out="$1" ctx_urls gen_urls n_ctx n_gen
  ctx_urls="$(build_urls ctx)"; gen_urls="$(build_urls gen)"
  # urls 개수 == num_instances 사전 검증 (orchestrator도 잡지만 더 빠르고 명확한 에러)
  n_ctx=$(awk 'NF{c++} END{print c+0}' <<< "$ctx_urls")
  n_gen=$(awk 'NF{c++} END{print c+0}' <<< "$gen_urls")
  [[ "$n_ctx" -eq "$NUM_CTX" ]] || { echo "[ERROR] ctx urls($n_ctx) != NUM_CTX($NUM_CTX)" >&2; exit 1; }
  [[ "$n_gen" -eq "$NUM_GEN" ]] || { echo "[ERROR] gen urls($n_gen) != NUM_GEN($NUM_GEN)" >&2; exit 1; }
  {
    echo "# auto-generated by launch_trtllm.sh — LABEL=$LABEL ($(date -u +%FT%TZ 2>/dev/null || echo gen))"
    echo "# NOTE(URL모드): 아래 tensor/pipeline_parallel_size·cache_transceiver_config는 정보용."
    echo "#   실제 워커 병렬화=CLI --tp_size/--pp_size, 실제 KV백엔드=워커 extra YAML(ctx/gen_*.yaml)."
    echo "hostname: $PROXY_HOST       # orchestrator 바인드 (CLI 불가 → 여기 필수)"
    echo "port: $PROXY_PORT           # sweep.py 조준 포트"
    echo "model: $MODEL               # ctx/gen에 상속 (get_llm_args positional model)"
    echo "backend: $BACKEND"
    echo "max_retries: 1"
    echo "perf_metrics_max_requests: $PERF_METRICS_MAX_REQUESTS   # orchestrator perf 버퍼(KV전송시간 /perf_metrics 노출). 0=off"
    echo "context_servers:"
    echo "  num_instances: $NUM_CTX"
    echo "  tensor_parallel_size: $CTX_TP"
    echo "  pipeline_parallel_size: $CTX_PP"
    echo "  cache_transceiver_config:"
    echo "    backend: $CACHE_BACKEND"
    echo "  urls:"
    while IFS= read -r u; do [[ -n "$u" ]] && echo "    - \"$u\""; done <<< "$ctx_urls"
    echo "generation_servers:"
    echo "  num_instances: $NUM_GEN"
    echo "  tensor_parallel_size: $GEN_TP"
    echo "  pipeline_parallel_size: $GEN_PP"
    echo "  cache_transceiver_config:"
    echo "    backend: $CACHE_BACKEND"
    echo "  urls:"
    while IFS= read -r u; do [[ -n "$u" ]] && echo "    - \"$u\""; done <<< "$gen_urls"
  } > "$out"
  echo "[gen] disagg config → $out"
}

# =============================================================================
# 역할 디스패치
# =============================================================================
ROLE="${1:-}"
[[ -z "$ROLE" ]] && { echo "Usage: $0 {context|generation|proxy|all}"; exit 1; }

PIDS=()

start_context() {
  local i port gpus
  for (( i=0; i<NUM_CTX; i++ )); do
    port=$(( CTX_PORT_BASE + i ))
    gpus="$(gpu_slice "$CTX_GPU_BASE" "$i" "$ctx_ranks")"
    PIDS+=("$(launch_worker context "$CTX_TP" "$CTX_PP" "$port" "$gpus" "$CTX_EXTRA")")
  done
}

start_generation() {
  local i port gpus
  for (( i=0; i<NUM_GEN; i++ )); do
    port=$(( GEN_PORT_BASE + i ))
    gpus="$(gpu_slice "$GEN_GPU_BASE" "$i" "$gen_ranks")"
    PIDS+=("$(launch_worker generation "$GEN_TP" "$GEN_PP" "$port" "$gpus" "$GEN_EXTRA")")
  done
}

start_proxy() {
  local cfg="$LOG_DIR/disagg_${LABEL}.yaml"
  write_disagg_yaml "$cfg"
  echo "[launch] orchestrator → ${PROXY_HOST}:${PROXY_PORT}  (cfg=$cfg)"
  # exec 안 씀 — 'all' 모드의 EXIT trap(워커 정리)이 동작하도록 포그라운드 실행
  trtllm-serve disaggregated \
    -c "$cfg" \
    -t "$SERVER_START_TIMEOUT" \
    -r "$REQUEST_TIMEOUT" \
    -l "$LOG_LEVEL" \
    2>&1 | tee "$LOG_DIR/trtllm_${LABEL}_proxy_$(hostname).log"
}

case "$ROLE" in
  context)    start_context;    echo "[pids] ${PIDS[*]:-}"; wait ;;
  generation) start_generation; echo "[pids] ${PIDS[*]:-}"; wait ;;
  proxy)      start_proxy ;;
  all)
    # 빈 배열 확장 가드(bash<4.4 이식성): start_context 실패로 PIDS가 비어도 안전
    trap 'echo "[trap] killing workers ${PIDS[*]:-}"; [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null || true' EXIT
    start_context
    start_generation
    echo "[pids] workers=${PIDS[*]:-}"
    # 워커 health 대기 (localhost 기준; inter-node면 proxy의 -t가 대기 처리)
    for (( i=0; i<NUM_CTX; i++ )); do wait_health localhost "$(( CTX_PORT_BASE + i ))" || exit 1; done
    for (( i=0; i<NUM_GEN; i++ )); do wait_health localhost "$(( GEN_PORT_BASE + i ))" || exit 1; done
    start_proxy
    ;;
  *) echo "Usage: $0 {context|generation|proxy|all}"; exit 1 ;;
esac
