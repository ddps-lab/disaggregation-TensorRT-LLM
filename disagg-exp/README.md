# disagg-exp — TRT-LLM PD 분리 실험 하네스 (실행 가이드)

> **무엇**: TensorRT-LLM v1.2.1로 Prefill/Decode를 분리(PD disaggregation)해, prefill·decode의 병렬화(TP·PP)를
> 대칭/비대칭으로, 토폴로지를 xPyD로 바꿔가며 TTFT/TPOT/throughput/$를 측정하는 하네스.
> **왜 이 폴더**: vLLM에서 시작했으나 PD에서 PP가 안 돼 TRT-LLM으로 이전. 1세대(vLLM) 방법론을 계승.
> 설계 배경/근거 = `CLAUDE.md`, 개념 = `LEARNING_NOTES.md`, 단계별 런북 = `EXPERIMENT_PLAN.md`.

---

## 1. 파일별 역할 — vLLM 1세대 대비 (있던거 / 새거 / 사라진거)

| 파일 | 역할 | vLLM 1세대 대비 |
|---|---|---|
| **`sweep.py`** | 부하 생성기. OpenAI `/v1/completions`에 token-id prompt를 Poisson으로 쏘고, SSE 스트림으로 TTFT/E2E 직접 측정. 2-phase(warmup→measured), `.done/.failed` resume, S3 자동 백업. | 🔄 **계승(거의 그대로)**. OpenAI 호환이라 프레임워크 무관. 바뀐 곳: `MODEL_NAME`→Qwen3-4B, metadata에 ctx/gen TP·PP·placement 기록. |
| **`analyze.py`** | 결과 JSONL→TTFT/TPOT/throughput(2종)/$per-Mtok 계산·표·플롯. | 🔄 **계승(로직 0 변경)**. 바뀐 곳: `COST_PER_HR`(g5/g6/g6e 단가), config 라벨(T1~T4). |
| **`setup.sh`** | 노드 부트스트랩. 측정 수집기(nvidia-smi dmon·ifstat·DCGM·chrony)·s5cmd 기동, 버전 검증. | 🔄 **계승(수집기 그대로)** + 🆕 **설치부 교체**: vLLM/LMCache/venv 빌드 → **NGC 컨테이너 모델**(tensorrt_llm 사전설치 확인). |
| **`launch_trtllm.sh`** | 서버 기동기. config+role(context/generation/proxy)별로 `trtllm-serve`를 띄우고 disagg config YAML을 런타임 생성. | 🆕 **신규** (vLLM의 `launch_configs.sh`를 대체). 골격(role 분기·env·포트규약)만 계승, 내부는 전면 교체. |
| **`ctx_extra_llm_api_options.yaml`** | **context(prefill) 워커**의 변인통제 (bf16·TRTLLM attn·block_reuse off·chunked off·cuda_graph off·overlap off). | 🆕 **신규**. vLLM에선 CLI 플래그(`--no-enable-prefix-caching` 등)였던 걸 TRT-LLM은 이 YAML로. |
| **`gen_extra_llm_api_options.yaml`** | **generation(decode) 워커**의 변인통제 (cuda_graph on·overlap on). | 🆕 **신규**. (동상) |
| **`disagg_config.yaml`** | orchestrator(=`trtllm-serve disaggregated`)가 읽는 **1P1D 정적 템플릿** (수동 실행/참조용). | 🆕 **신규** (vLLM의 `disagg_proxy_server.py`를 대체하는 개념). |
| **`trtllm_support_matrix.md`** | **Phase 0 게이트** 결과표 — 어떤 (TP,PP) 조합이 안 깨지나 실측 기록. | 🆕 **신규** (TRT-LLM은 비대칭 PP가 미보증이라 사전 게이트 필요). |

**❌ vLLM에 있었지만 안 가져온 것 (TRT-LLM에 불필요):**
- `launch_configs.sh` → `launch_trtllm.sh`로 대체
- `disagg_proxy_server.py` → `trtllm-serve disaggregated`(내장 orchestrator)로 대체
- `instrumented_connector.py` (LMCache KVConnector 계측) → TRT-LLM은 자체 KV전송(cache_transceiver). KV전송 시간은 orchestrator `/perf_metrics`로.
- LMCache YAML / `--kv-transfer-config` → TRT-LLM `cache_transceiver_config`(UCX)로 대체.

> **요약**: 측정·분석(sweep/analyze/setup의 수집기)은 vLLM 그대로 계승, **서버 기동·KV전송 부분만 TRT-LLM 네이티브로 교체.**

---

## 2. 어떻게 돌리나

### 사전: 컨테이너 (모든 런타임은 원격 GPU에서)
```bash
docker run --rm -it --gpus all --network host --ipc host --shm-size=8g \
    -v "$PWD":/work -w /work/disagg-exp \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 bash
# (컨테이너 안에서)
bash setup.sh          # tensorrt_llm 1.2.1 확인 + 수집기·deps 설치
```

### 0) smoke — 본 스윕 전 빠른 검증 (디버깅용, 실제 config + 적은 요청)
"길게 켜놓고 에러나면 비싸다" → 같은 launch로 띄우고 요청만 적게 5분 검증 후 본 스윕.
```bash
# 서버(디버그 로그 ON): 실제 쓸 config 그대로
LABEL=smoke NUM_CTX=1 NUM_GEN=1 CTX_TP=1 CTX_PP=1 GEN_TP=1 GEN_PP=1 LOG_LEVEL=debug bash launch_trtllm.sh all
# 스윕(단일 포인트, warmup 3 / measured 5):
SWEEP_PD_PAIRS="1024,512" SWEEP_RATES=1.0 SWEEP_WARMUP_N=3 SWEEP_MEASURED_N=5 \
  python sweep.py --config smoke --base-url http://localhost:8000 --s3-bucket ""
python analyze.py --configs smoke      # status:success + kv_p50ms 컬럼에 값 → 통과
# 통과하면 LOG_LEVEL 빼고(=info) 아래 본 스윕. (디버그 토글 상세 → DEBUGGING.md)
```

### A) 단일 노드 (Phase 0 스파이크 — 권장 시작점, g6e.12xlarge=4×L40S)
**셸 1 — 서버 기동** (ctx+gen+orchestrator 한 번에):
```bash
LABEL=T1 NUM_CTX=1 NUM_GEN=1 \
CTX_TP=1 CTX_PP=1 GEN_TP=1 GEN_PP=1 \
bash launch_trtllm.sh all
#  → ctx 워커(GPU0,:8001) + gen 워커(GPU1,:8011) + orchestrator(:8000)
#  → /health 통과까지 기다렸다 orchestrator 포그라운드 실행
```
**셸 2 — 스윕** (orchestrator :8000 조준):
```bash
# ⚠️ 서버와 같은 토폴로지 env를 export 해야 metadata.json이 정확히 기록됨
export LABEL=T1 NUM_CTX=1 NUM_GEN=1 CTX_TP=1 CTX_PP=1 GEN_TP=1 GEN_PP=1 PLACEMENT=intra
python sweep.py --config T1 --base-url http://localhost:8000
```
**셸 3 (스윕 후) — 분석**:
```bash
python analyze.py --configs T1 --plot
```

**비대칭/xPyD 예** (env만 바꾸면 됨):
```bash
# 1P3D, decode 3개 복제:        NUM_GEN=3
# 비대칭 TP (P tp2 → D tp1):    CTX_TP=2 GEN_TP=1   (≥3 GPU 필요)
# 대칭 PP:                       CTX_PP=2 GEN_PP=2   (≥4 GPU)
# ⚠️ ctx-PP→gen-TP (CTX_PP=2 GEN_TP=2): hang #14020 위험 → 300s 감시
```

### B) Inter-node (노드 분리)
```bash
# 노드 A (context):    LABEL=T1 NUM_CTX=1 CTX_TP=1 CTX_PP=1 CTX_GPU_BASE=0 bash launch_trtllm.sh context
# 노드 B (generation): LABEL=T1 NUM_GEN=1 GEN_TP=1 GEN_PP=1 GEN_GPU_BASE=0 bash launch_trtllm.sh generation
# 노드 C (orchestrator; A에서 겸해도 됨):
LABEL=T1 NUM_CTX=1 NUM_GEN=1 PLACEMENT=inter \
CTX_URLS="<A_사설IP>:8001" GEN_URLS="<B_사설IP>:8011" \
bash launch_trtllm.sh proxy
# 스윕: python sweep.py --config T1 --base-url http://<C_IP>:8000
```
> inter-node는 EFA 없으면 KV전송이 TCP라 느림 → 구조/correctness 비교용. 깨끗한 성능은 intra-node 중심.

---

## 3. 주요 환경변수 (launch_trtllm.sh)

| 변수 | 기본 | 의미 |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-4B` | positional 모델 (HF id) |
| `LABEL` | `T1` | config 라벨 (sweep `--config`·analyze `COST_PER_HR` 키와 일치시킬 것) |
| `NUM_CTX` / `NUM_GEN` | `1` / `1` | xPyD의 P / D 개수 (1P3D면 `NUM_GEN=3`) |
| `CTX_TP` `CTX_PP` / `GEN_TP` `GEN_PP` | `1` | prefill / decode 병렬화 (비대칭축) |
| `CACHE_BACKEND` | `UCX` | KV전송: DEFAULT\|UCX\|NIXL\|MOONCAKE\|MPI (ctx==gen) |
| `CTX_GPU_BASE` / `GEN_GPU_BASE` | `0` / `ctx 뒤` | GPU 배치 시작 인덱스 (inter면 각 노드서 0) |
| `CTX_PORT_BASE` / `GEN_PORT_BASE` / `PROXY_PORT` | `8001`/`8011`/`8000` | 포트 규약 (proxy 8000 = sweep 조준, 고정) |
| `CTX_URLS` / `GEN_URLS` | (localhost 자동) | inter-node 워커 host:port 목록 (개수==NUM_CTX/GEN) |
| `EXP_LOG_DIR` | `./results` | 결과·생성 YAML·로그 경로 |
| `USE_SERVER_ROLE` | `0` | 1이면 `--server_role CONTEXT/GENERATION` 추가 (Phase 0서 필요 판명 시) |
| `LOG_LEVEL` | `info` | 서버 로그레벨(워커 `--log_level`/orchestrator `-l`). **측정=info, 디버그=debug** (→ DEBUGGING.md) |
| `PERF_METRICS_MAX_REQUESTS` | `1000` | orchestrator perf 버퍼(KV전송시간 `/perf_metrics`). 0=off |

**sweep.py env**: `SWEEP_PD_PAIRS`("2048,128;1024,512", smoke용 단일포인트) · `SWEEP_RATES` · `SWEEP_WARMUP_N`(기본 20, smoke 3) · `SWEEP_MEASURED_N`(300) · `PLACEMENT`(intra|inter, metadata용).

---

## 4. 트러블슈팅 (코드 검증으로 미리 막은 함정)
- **워커 기동 즉시 ValidationError**: extra YAML 키 오타. `enable_block_reuse`/`free_gpu_memory_fraction`는 **반드시 `kv_cache_config:` 하위** (StrictBaseModel extra=forbid).
- **`--model`/`--dtype`/`--served-model-name` 에러**: 그런 CLI 플래그 **없음**. 모델은 positional, dtype은 extra YAML.
- **orchestrator가 포트 무시**: orchestrator host/port는 CLI 불가 → disagg YAML의 `hostname`/`port`에서 읽음 (launch_trtllm.sh가 자동 생성).
- **proxy `/metrics` 404**: orchestrator엔 `/metrics` 없음 → 워커(:8001 등) `/metrics` 또는 proxy `/perf_metrics` 사용.
- **요청이 EOS에서 조기 종료**: `ignore_eos`+`max_tokens`로 길이 강제 (v1.2.1 지원 확인). 필요시 payload에 `min_tokens` 추가.
- **hang (무응답 300s+)**: ctx-PP→gen-TP 조합(#14020) 의심 → `trtllm_support_matrix.md`에 FAIL 기록, 조합 제외 or 1.3.0rc 검토.

---

## 5. 산출물 & 디버깅
**측정 산출물** (`$EXP_LOG_DIR/<config>/`):
- `p{..}.jsonl` — 요청별 TTFT/E2E/status (sweep)
- `perf_p{..}.json` — `/perf_metrics` 스냅샷(KV전송시간·블록재사용) ★
- `metadata.json` — ctx/gen TP·PP·xPyD·placement·cache_backend
- 시스템: `nvidia_smi.csv`·`ifstat.csv`·`dcgm.log` (1Hz/2s), `s3_sync.log`, `clock_baseline_*`

**분석**: `analyze.py --plot` → TTFT/TPOT/throughput(2종)/$Mtok **+ `kv_p50ms`/`kv_p99ms`(KV전송시간)** 표·플롯.

**디버깅 / 로그구조 / KV·perf 읽는 법 / hang 진단 / 디버그 토글 켜고 끄기 → `DEBUGGING.md`.**
> ⚠️ 측정 런에선 디버그 로그 OFF(`LOG_LEVEL` 미설정). 디버그 로그 = I/O 노이즈 → 변인 오염.
