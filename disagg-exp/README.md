# disagg-exp — TRT-LLM PD 분리 실험 하네스 (실행 가이드)

> **무엇**: TensorRT-LLM v1.2.1로 Prefill/Decode를 분리(PD disaggregation)해, prefill·decode의 병렬화(TP·PP)를
> 대칭/비대칭으로, 토폴로지를 xPyD로 바꿔가며 TTFT/TPOT/throughput/$를 측정하는 하네스.
> **왜 이 폴더**: vLLM에서 시작했으나 PD에서 PP가 안 돼 TRT-LLM으로 이전. 1세대(vLLM) 방법론을 계승.
> 설계 배경/근거 = `CLAUDE.md`, 개념 = `LEARNING_NOTES.md`, 단계별 런북 = `EXPERIMENT_PLAN.md`, 디버깅 = `DEBUGGING.md`.
> **전체 문서가 각각 뭐하는지 = `CLAUDE.md` 파일맵 표**(8종 역할·언제 여나 정리).

---

## 1. 파일별 역할 — vLLM 1세대 대비 (있던거 / 새거 / 사라진거)

| 파일 | 역할 | vLLM 1세대 대비 |
|---|---|---|
| **`sweep.py`** | **오케스트레이터**. 그리드 루프마다 **공식 `benchmark_serving`을 서브프로세스로 호출**(warmup→measured → `bench_<point>.json`), measured 윈도우 전/후 per-side 스냅샷, `.done/.failed` resume, S3 자동 백업, metadata. | 🔄 골격 계승 + 🆕 **부하코어 교체**: 손수 짠 aiohttp → **공식 benchmark_serving**(측정=공식, sweep=오케스트레이션만). |
| **`prom_scrape.py`** | per-side 스크레이퍼. orchestrator `/prometheus/metrics`의 `ctx_/gen_completed_requests_total`를 스냅샷 → prefill/decode RPS. | 🆕 **신규** (공식이 per-side를 안 줌 — 유일한 커스텀 메트릭). |
| **`analyze.py`** | `bench_<point>.json`(공식 집계 TTFT/TPOT/ITL/E2EL/throughput) + `prom_*`(per-side) + `perf_*`(KV) → 표·플롯·$per-Mtok. | 🔄 **공식 result.json 읽기로 전환**(우리 계산 제거). `COST_PER_HR`(g5/g6/g6e), config T1~T4. |
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
python analyze.py --configs smoke      # bench_*.json 생성 + 표에 ttft/tpot/e2el 값 + (가능하면) pf/dc_rps → 통과
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

### B-0) inter-node P1D1 smoke (g6.xlarge ×2 — 첫 런타임 검증, 권장 시작점)
> 1-GPU 박스 2대로 P1D1: 노드 P=prefill(ctx 워커 + orchestrator), 노드 D=decode(gen 워커).
> "길게 켜기 전 적은 요청 5분 검증". 모든 런타임은 원격(SSH), 로컬은 편집·git만(aws 규칙).

**0. 사전 (양 노드 공통, 컨테이너 안)**
```bash
# (IP 갱신/host key 바뀌었으면) 로컬에서: ssh-keygen -R <공인IP> + ~/.ssh/config HostName 수정
git clone https://github.com/ddps-lab/disaggregation-TensorRT-LLM.git
cd disaggregation-TensorRT-LLM && git checkout disagg-exp/trtllm-v1.2.1
docker run --rm -it --gpus all --network host --ipc host --shm-size=8g \
  -v "$PWD":/work -w /work/disagg-exp nvcr.io/nvidia/tensorrt-llm/release:1.2.1 bash
bash setup.sh                              # 1.2.1 확인 + 수집기 + chrony(시계동기, inter 필수)
hostname -I | awk '{print $1}'             # 각 노드 사설IP 확보 (orchestrator가 씀)
```
**⚠️ 보안그룹**: 두 노드가 `:8001`/`:8011` + UCX(에페메랄 포트)로 통신 → 같은 VPC/서브넷 + SG가 노드 간 트래픽 허용(self-referencing SG로 전부 열기). 막히면 health/KV전송 멈춤.

**1. 노드 D (decode) 먼저**
```bash
export UCX_TLS=tcp,cuda_copy,sm,self       # cross-node = TCP 강제(EFA 없음). 안 주면 UCX 실패 가능
LABEL=smoke NUM_GEN=1 GEN_TP=1 GEN_PP=1 GEN_GPU_BASE=0 LOG_LEVEL=debug \
  bash launch_trtllm.sh generation         # :8011
```
**2. 노드 P (context 워커 + orchestrator)**
```bash
export UCX_TLS=tcp,cuda_copy,sm,self
LABEL=smoke NUM_CTX=1 CTX_TP=1 CTX_PP=1 CTX_GPU_BASE=0 LOG_LEVEL=debug \
  bash launch_trtllm.sh context            # :8001
LABEL=smoke NUM_CTX=1 NUM_GEN=1 PLACEMENT=inter \
  CTX_URLS="<P_사설IP>:8001" GEN_URLS="<D_사설IP>:8011" \
  bash launch_trtllm.sh proxy              # :8000 ← sweep 조준점
```
**3. 부하(적은 요청) + 분석 (노드 P, 새 셸)**
```bash
SWEEP_PD_PAIRS="1024,512" SWEEP_RATES=1.0 SWEEP_WARMUP_N=3 SWEEP_MEASURED_N=5 PLACEMENT=inter \
  python sweep.py --config smoke --base-url http://localhost:8000 --s3-bucket ""
python analyze.py --configs smoke
```
**4. per-side 카운터 실노출 확인**
```bash
curl -s localhost:8000/prometheus/metrics | grep -E 'ctx_completed|gen_completed'
cat results/smoke/prom_p1024_d512_r1.0.json
```
**통과 기준**: 기동 `[ok] healthy` ×3 → sweep가 `bench_*.json`(공식 결과)·`perf_*.json`·`prom_*.json` 생성 → analyze 표에 `ttft/tpot/e2el`·`kv_p50ms` 값 → `ctx_/gen_completed_requests_total` **둘 다 >0**(+ 표 `pf_rps/dc_rps/pf_tps/dc_tps`). 카운터 안 보이면 graceful `n/a` → 워커 `trtllm_request_success_total` 폴백(예상된 미지수, `병렬화-KV전송-측정.md §9`).
**inter 특유 실패**: health timeout=SG 미개방 / bench rc≠0·hang=KV전송 멈춤(UCX_TLS·SG) / kv n/a=`return_perf_metrics` 확인. KV전송 느림(큰 TTFT)은 TCP라 정상.

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
- `bench_p{..}.json` — **공식 benchmark_serving 결과** (TTFT/TPOT/ITL/E2EL/throughput + per-request ITL) ★
- `perf_p{..}.json` — `/perf_metrics` 스냅샷(KV전송시간·블록재사용) ★
- `prom_p{..}.json` — measured 윈도우 전/후 per-side 카운터 스냅샷(prefill/decode RPS 산출용) ★
- `metadata.json` — ctx/gen TP·PP·xPyD·placement·cache_backend
- 시스템: `nvidia_smi.csv`·`ifstat.csv`·`dcgm.log` (1Hz/2s), `s3_sync.log`, `clock_baseline_*`

**분석**: `analyze.py --plot` → **공식 집계** `ttft_p50/p99`·`tpot_p50/p99`·`itl_p99`·`e2el_p99`·`out_tok/s`·`$/Mtok` + `kv_p50/p99`(KV전송시간) **+ `pf_rps`/`dc_rps`/`pf_tps`/`dc_tps`(per-side)** 표·플롯. (kv·per-side는 perf/prom 있을 때만, 없으면 'n/a')

**디버깅 / 로그구조 / KV·perf 읽는 법 / hang 진단 / 디버그 토글 켜고 끄기 → `DEBUGGING.md`.**
> ⚠️ 측정 런에선 디버그 로그 OFF(`LOG_LEVEL` 미설정). 디버그 로그 = I/O 노이즈 → 변인 오염.
