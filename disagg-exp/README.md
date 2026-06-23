# disagg-exp — TRT-LLM PD 분리 실험 하네스 (실행 가이드)

> **무엇**: TensorRT-LLM v1.2.1로 Prefill/Decode를 분리(PD disaggregation)해, prefill·decode의 병렬화(TP·PP)를
> 대칭/비대칭으로, 토폴로지를 xPyD로 바꿔가며 TTFT/TPOT/throughput/$를 측정하는 하네스.
> **왜 이 폴더**: vLLM에서 시작했으나 PD에서 PP가 안 돼 TRT-LLM으로 이전. 1세대(vLLM) 방법론을 계승.
> 설계 배경/근거 = `CLAUDE.md`, 개념·프레임워크판정·병렬화조사 = `LEARNING_NOTES.md`, 실험순서·변인통제·디버깅 = 이 `README.md`(아래 §실험 순서 / §변인 통제 / §디버깅), 작업기록·Phase 0 결과 = `SETUP_LOG.md`.
> **전체 문서 인덱스 = `CLAUDE.md §문서 인덱스`**(4종 역할 정리).

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
| disagg orchestrator config | orchestrator(=`trtllm-serve disaggregated -c`)가 읽는 config. **`launch_trtllm.sh`가 `write_disagg_yaml()`로 런별 런타임 생성**(`$LOG_DIR/disagg_<LABEL>.yaml`) — 정적 파일 아님. | 🆕 **신규** (vLLM의 `disagg_proxy_server.py`를 대체하는 개념). |
| Phase 0 지원 매트릭스 | **Phase 0 게이트** 결과표 — 어떤 (TP,PP) 조합이 안 깨지나 실측 기록. 이제 `SETUP_LOG.md §Phase 0 지원 매트릭스`에 있음. | 🆕 **신규** (TRT-LLM은 비대칭 PP가 미보증이라 사전 게이트 필요). |

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
# 통과하면 LOG_LEVEL 빼고(=info) 아래 본 스윕. (디버그 토글 상세 → README.md §디버깅)
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
**통과 기준**: 기동 `[ok] healthy` ×3 → sweep가 `bench_*.json`(공식 결과)·`perf_*.json`·`prom_*.json` 생성 → analyze 표에 `ttft/tpot/e2el`·`kv_p50ms` 값 → `ctx_/gen_completed_requests_total` **둘 다 >0**(+ 표 `pf_rps/dc_rps/pf_tps/dc_tps`). 카운터 안 보이면 graceful `n/a` → 워커 `trtllm_request_success_total` 폴백(예상된 미지수, `LEARNING_NOTES.md §병렬화·KV전송·per-side` 9번).
**inter 특유 실패**: health timeout=SG 미개방 / bench rc≠0·hang=KV전송 멈춤(UCX_TLS·SG) / kv n/a=`return_perf_metrics` 확인. KV전송 느림(큰 TTFT)은 TCP라 정상.

---

## 3. 주요 환경변수 (launch_trtllm.sh)

| 변수 | 기본 | 의미 |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-4B` | positional 모델 (HF id) |
| `LABEL` | `T1` | config 라벨 (sweep `--config`·analyze `COST_PER_HR` 키와 일치시킬 것) |
| `NUM_CTX` / `NUM_GEN` | `1` / `1` | xPyD의 P / D 개수 (1P3D면 `NUM_GEN=3`) |
| `CTX_TP` `CTX_PP` / `GEN_TP` `GEN_PP` | `1` | prefill / decode 병렬화 (비대칭축) |
| `CACHE_BACKEND` | `NIXL` | KV전송: DEFAULT(=NIXL)\|UCX\|NIXL\|MOONCAKE\|MPI (ctx==gen). no-EFA→내부 UCX-TCP. 폴백 UCX |
| `CTX_GPU_BASE` / `GEN_GPU_BASE` | `0` / `ctx 뒤` | GPU 배치 시작 인덱스 (inter면 각 노드서 0) |
| `CTX_PORT_BASE` / `GEN_PORT_BASE` / `PROXY_PORT` | `8001`/`8011`/`8000` | 포트 규약 (proxy 8000 = sweep 조준, 고정) |
| `CTX_URLS` / `GEN_URLS` | (localhost 자동) | inter-node 워커 host:port 목록 (개수==NUM_CTX/GEN) |
| `EXP_LOG_DIR` | `./results` | 결과·생성 YAML·로그 경로 |
| `USE_SERVER_ROLE` | `0` | 1이면 `--server_role CONTEXT/GENERATION` 추가 (Phase 0서 필요 판명 시) |
| `LOG_LEVEL` | `info` | 서버 로그레벨(워커 `--log_level`/orchestrator `-l`). **측정=info, 디버그=debug** (→ README.md §디버깅) |
| `PERF_METRICS_MAX_REQUESTS` | `1000` | orchestrator perf 버퍼(KV전송시간 `/perf_metrics`). 0=off |

**sweep.py env**: `SWEEP_PD_PAIRS`("2048,128;1024,512", smoke용 단일포인트) · `SWEEP_RATES` · `SWEEP_WARMUP_N`(기본 20, smoke 3) · `SWEEP_MEASURED_N`(300) · `PLACEMENT`(intra|inter, metadata용).

---

## 실험 순서 / 로드맵

> *무엇을/왜/어떤 순서로* 측정하나 = 설계·런북. **핀·변인통제값·dealbreaker·disagg 스키마는 `CLAUDE.md`** — 여기선 실험축·Phase 0~3·매트릭스.

### 실험 축 (사용자 진행 순서 반영)
- **토폴로지(xPyD)**: 1P1D → **1P3D** → 확장(D 개수↑). decode가 throughput 병목이라 D 스케일이 핵심.
- **병렬화**: prefill (TP_p, PP_p), decode (TD_d, PP_d) — **대칭(P=D)·비대칭(P≠D)** 둘 다. 특히 "D에 여러 병렬화" = 각 decode 인스턴스의 TP/PP 변형.
- **배치**: inter-node 먼저 → intra-node.
- **측정**: 기존 하네스의 TTFT / TPOT / throughput(2종) / $-per-Mtok (analyze.py 그대로).

### 진행 시퀀스 (사용자 명시)
1. **inter-node 1P1D** (베이스라인)
2. **1P3D, decode 병렬화 다양화, inter-node** (대칭→비대칭)
3. **intra-node 가정으로 동일 스윕**
4. **D 확장**(D3→그 이상)

### Phase 0 — TRT-LLM 검증 스파이크 (게이트, multi-GPU 1대)

대량 스윕 전, "무엇이 안 깨지고 도는가"를 실측 매트릭스로 확정. (선행 조사에서 함정 식별됨)

1. **환경**: g6e.12xlarge(4×L40S) 또는 g6.12xlarge(4×L4) 1대. 컨테이너/버전 **통째 핀** — 시작점 **v1.2.1**(+ NIXL 버전 포함). `--backend pytorch`.
2. **단일(비분리) sanity**: `trtllm-serve Qwen3-4B --backend pytorch --tp_size 2`로 Qwen3-4B 로드+추론 확인 (Qwen3 지원 버전 확인).
3. **최소 disagg (1 ctx + 1 gen + orchestrator)**, intra-node, TCP 전송:
   ```bash
   # context
   CUDA_VISIBLE_DEVICES=0,1 trtllm-serve Qwen3-4B --backend pytorch --tp_size 2 --port 8001 \
     --extra_llm_api_options ctx.yaml   # cache_transceiver_config.backend: NIXL; disable_overlap_scheduler: True
   # generation
   CUDA_VISIBLE_DEVICES=2,3 trtllm-serve Qwen3-4B --backend pytorch --tp_size 2 --port 8002 \
     --extra_llm_api_options gen.yaml   # enable_block_reuse: false
   # orchestrator (OpenAI 호환 단일 엔드포인트)
   trtllm-serve disaggregated -c disagg.yaml   # context_servers / generation_servers URL 나열
   ```
4. **조합 매트릭스 실측** → `SETUP_LOG.md §Phase 0 지원 매트릭스`:
   - 대칭 TP: P(tp2)·D(tp2) / 비대칭 TP: P(tp1)·D(tp2), P(tp2)·D(tp1)
   - 대칭 PP: P(pp2)·D(pp2) / 비대칭 PP: P(pp2)·D(pp1), P(pp1)·D(pp2)
   - ⚠️ **ctx-PP→gen-TP(예: P pp2×tp1 → D pp1×tp2)는 알려진 hang(#14020)** → 300s hang_detector 감시. 막히면 1.3.0rc로 핀 변경 재시도.
   - 각 셀: 기동 성공 / 첫 요청 KV전송 성공 / **출력 정확성**(비분리 동일 seed와 비교, 과거 비대칭 TP garbled #6507 회귀 점검; **TRTLLM attention 백엔드 사용, FlashInfer 금지**).
5. **게이트**: 통과 조합으로만 Phase 1 매트릭스 구성. PP가 광범위하게 막히면 사용자와 재상의(버전 올림 vs 축 축소).

### Phase 2 — Config 매트릭스 (스파이크 통과분 위에서)

`(xPyD, P:[TP,PP], D:[TP,PP], placement)` 격자. 예시 시작점:
- T1: inter 1P1D, P tp1·D tp1 (대칭 베이스라인)
- T2: inter 1P3D, D 각각 tp1 (복제) → tp2 → pp2 (decode 병렬화 다양화)
- T3: 비대칭 — P tp2 → D tp1 / P tp1 → D tp2
- T4: intra 가정으로 T1~T3 반복 (PCIe 전송)
각 config는 inter/intra와 EFA 유무를 로그에 명시(전송 변수 분리).

### Phase 3 — 검증 (end-to-end)

1. Phase 0 게이트 통과.
2. Smoke: `launch_trtllm.sh <cfg> {context,generation,orchestrator}` → `/health` → `sweep.py --config smoke` 1 point → `bench_*.json` 생성(공식 결과).
3. 비대칭 TP 1 point에서 KV전송/출력정확성 sanity.
4. small sweep → full sweep (기존 `.done/.failed` resume).
5. `analyze.py --plot` — TTFT/TPOT/$Mtok + (xPyD, 병렬화 조합)별 비교.

---

## 변인 통제

> 이 실험은 `vllm-disaggregation/disagg-exp`의 방법론을 그대로 잇는다. **클라이언트 측(`sweep.py`)·메트릭(`analyze.py`)·2-phase·수집기·시계동기·S3는 프레임워크 무관 → 거의 그대로 재사용**, **서버 측 noise control만 TRT-LLM 플래그로 매핑**한다. (클라이언트·메트릭·인프라 통제의 정본 규칙 = `CLAUDE.md §변인통제`. 부하·측정 코어 = 공식 `benchmark_serving`.)

### 하드웨어 / 전송 현실 (EC2, 검증됨)

| 항목 | 사실 | 함의 |
|---|---|---|
| GPU | g5=A10G(SM86), g6=L4(SM89), g6e=L40S(SM89) | 전부 TRT-LLM OK. dtype: A10G/L4/L40S 모두 FP16/BF16(+FP8 Ada). 공정 비교는 **BF16 또는 FP16 통일** |
| NVLink | g5/g6/g6e **없음**(PCIe만) | intra-node KV전송 = PCIe. cuda_ipc는 Nitro에서 막힐 수 있음 → UCX가 shm/cuda_copy로 처리 |
| EFA | xlarge·12xlarge 대부분 **없음** (최상위 사이즈만) | **inter-node KV전송 = TCP(ENA) → 느림.** inter-node는 구조/correctness용, 깨끗한 성능 결론은 intra-node 위주 |
| 전송 백엔드 | TRT-LLM 기본 UCX, TCP fallback | `cache_transceiver_config.backend: UCX` 고정. Ethernet 노드 `UCX_TLS=tcp,cuda_copy,sm,self` |

### ① 서버측 변인 통제 (vLLM → TRT-LLM 매핑)
| 통제 의도 | vLLM (기존) | TRT-LLM 대응 |
|---|---|---|
| prefix/KV 재사용 차단(캐시 오염 방지) | `--no-enable-prefix-caching` | `kv_cache_config.enable_block_reuse: false` (PP hang 회피에도 필요) |
| prefill 한 번에(compute-bound 측정) | `--no-enable-chunked-prefill` (PD prefill) | context 서버 `enable_chunked_prefill: false` |
| dtype 통일(cross-config 왜곡 방지) | `--dtype half` | **`--dtype bfloat16` 통일** — A10G/L4/L40S 전부 native(T4 빠져 fp16 강제 불필요) |
| 큐 병목 제거 | `--max-num-seqs 512` | `--max_batch_size`(+`max_num_tokens`) 상향 |
| CUDA Graph ON | enforce-eager 안 씀 | PyTorch 백엔드 `cuda_graph_config` ON(decode) |
| KV 메모리 비율 | `--gpu-memory-utilization 0.85` | `kv_cache_config.free_gpu_memory_fraction: 0.85` |
| 분산 재현성 | `PYTHONHASHSEED=123` | 동일 + 결정론 샘플링(temp=0) |

> ⚠️ CUDA Graph는 위 표가 vLLM→TRT 매핑의 출발점이나, **본 실험은 양쪽 `cuda_graph_config: null`로 OFF**(균일 eager) — 사용자 결정(변인통제 + graph-capture cold-start 제거). 정본 = `CLAUDE.md §변인통제`.

### ⑤ 워크로드 그리드 (계승 + Qwen3-4B/하드웨어 맞춰 조정)
- **PD_PAIRS**(prefill,decode): `(2048,128)` prefill-heavy · `(1024,512)` balanced · `(128,2048)` decode-heavy — 3종 유지.
- **RATES**: `[1.0, 2.0, 4.0]`(저/중/포화). saturation 보려면 상향 가능.
- **max-model-len**: T4용 4096 제약 **해제**(L4 24GB/L40S 48GB) → 더 긴 조합 가능하나 비교 위해 고정값 명시.
- 각 config의 `metadata.json`에 **P/D의 (TP,PP) + xPyD(P·D 수) + placement(inter/intra) + EFA 유무** 기록(기존 `tp_map/pp_map` 패턴 확장) → 전송 변수 분리.

---

## 디버깅

> ⚠️ **측정(measured) 런에선 모든 디버그 로그를 OFF로.** 디버그 로그는 I/O 노이즈 → 변인 오염(research-rigor #7).
> 디버깅은 **smoke(실제 config + 적은 요청)** 에서만 켜고, 통과하면 끄고 본 스윕.

### 빠른 디버깅 흐름 (smoke)
"길게 켜놓고 에러나면 비싸다" → **실제 config 그대로, 요청만 적게** 5분 검증:
```bash
# 1) 실제 config로 서버 기동 (디버그 로그 ON)
LABEL=smoke NUM_CTX=1 NUM_GEN=1 CTX_TP=1 CTX_PP=1 GEN_TP=1 GEN_PP=1 \
LOG_LEVEL=debug bash launch_trtllm.sh all
# 2) 최소 요청 (단일 PD 포인트, warmup 3 / measured 5)
SWEEP_PD_PAIRS="1024,512" SWEEP_RATES=1.0 SWEEP_WARMUP_N=3 SWEEP_MEASURED_N=5 \
python sweep.py --config smoke --base-url http://localhost:8000 --s3-bucket ""
# 3) 확인: bench_*.json 생성(공식 결과) + perf에 KV전송시간 존재
python analyze.py --configs smoke
```
통과하면 → `LOG_LEVEL` 빼고(=info) 본 스윕.

### 로그 레벨 켜고 끄기

| 켜는 법 | 무엇 | 끄는 법 |
|---|---|---|
| `LOG_LEVEL=debug bash launch_trtllm.sh ...` | 워커 `--log_level` + orchestrator `-l` 한 번에 | `LOG_LEVEL` 생략(=info) |
| `export TLLM_LOG_LEVEL=debug` | TRT-LLM 파이썬 상세 로그(debug\|verbose\|trace) | `unset TLLM_LOG_LEVEL` |
| `export UCX_LOG_LEVEL=debug` | UCX(KV전송 전송계층) 상세 로그 | `unset UCX_LOG_LEVEL` |

`launch_trtllm.sh` 상단 **DEBUG 토글 블록**에 위 env가 주석으로 박혀 있음(주석 해제=ON, 재주석=OFF). 각 줄 끝에 "무슨 디버그인지" 명시.

### 로그 구조 (어디에 뭐가 있나) — `$EXP_LOG_DIR/` (기본 `./results`)

```
results/
├── trtllm_<LABEL>_context_p8001_<host>.log   # context 워커 stdout/stderr
├── trtllm_<LABEL>_generation_p8011_<host>.log # generation 워커
├── trtllm_<LABEL>_proxy_<host>.log            # orchestrator (tee)
├── disagg_<LABEL>.yaml                        # 런타임 생성된 orchestrator config
├── nvidia_smi.csv      ifstat.csv   dcgm.log  # 1Hz/2s 시스템 메트릭 (setup.sh 수집기)
├── clock_baseline_<host>.txt  s3_sync.log
├── .pid_nvidia_dmon  .pid_ifstat  .pid_dcgm_loop
└── <config>/                                  # 예: T1/, smoke/
    ├── bench_p{pl}_d{dl}_r{rate}.json         # 공식 benchmark_serving 결과 (TTFT/TPOT/ITL/E2EL/throughput)
    ├── prom_p{pl}_d{dl}_r{rate}.json          # per-side 카운터 스냅샷 (prefill/decode RPS)
    ├── perf_p{pl}_d{dl}_r{rate}.json          # /perf_metrics 스냅샷 (KV전송시간 등)
    ├── .done_* / .failed_*                    # resume 마커
    └── metadata.json                          # config·ctx/gen TP·PP·placement·load_tool
```

**기동 에러 추적**: 워커가 즉시 죽으면 `trtllm_<LABEL>_context_*.log` 끝부분 확인 (대부분 extra YAML 키 오타 → `ValidationError`).

### KV전송 시간 / perf 메트릭 읽기

활성(기본 ON): disagg YAML top-level `perf_metrics_max_requests: 1000` + 워커 extra YAML `return_perf_metrics: true`.

- **수집**: `sweep.py`가 각 포인트 측정 종료 후 orchestrator `GET /perf_metrics` 1회 → `<config>/perf_p..._.json`.
- **분석**: `analyze.py`가 자동으로 읽어 표에 `kv_p50ms`/`kv_p99ms` 컬럼 출력.
- **수동 확인**:
  ```bash
  curl -s http://localhost:8000/perf_metrics | python -m json.tool | head -50
  ```
  per-request KV전송시간 = `gen_perf_metrics[.perf_metrics].timing_metrics.kv_cache_transfer_end - ..._start` (초, **`kv_cache_size>0`일 때만 존재**).
  > ⚠️ `timing_metrics`가 `perf_metrics` 래퍼 아래 중첩되는지(`gen_perf_metrics.perf_metrics.timing_metrics`)는 버전차 가능 — sweep/analyze 파서는 **둘 다 자동 대응**. smoke에서 `curl .../perf_metrics`로 실제 키 깊이 한 번 눈으로 확인 권장.
- **주의**: 응답은 FIFO 리스트(요청 ID 없음). orchestrator엔 `/metrics` 없음(404) → 워커 `/metrics` 또는 이 `/perf_metrics` 사용.

### disagg 전용 디버그 env (각 무엇 / 끄는 법)

| env | 무엇을 하나 | 끄는 법 |
|---|---|---|
| `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` | KV전송 overlap 비활성 → 전송 타이밍 단순화(분리 측정) | `unset` |
| `UCX_TLS=tcp,cuda_copy,sm,self` | UCX 전송 경로 강제(EFA 없음 노드) | 기본값이라 보통 유지 |
| `UCX_LOG_LEVEL=debug` | UCX 핸드셰이크/전송 상세 | `unset` |
| `CACHE_BACKEND=NIXL` (launch env) | KV 백엔드 — **단 실제값은 워커 extra YAML이 결정**(disagg YAML은 정보성) | extra YAML 수정 |

> 전송 백엔드를 정말 바꾸려면 `ctx/gen_extra_llm_api_options.yaml`의 `cache_transceiver_config.backend`를 직접 수정.

### hang (무응답 300s+) 진단 — #14020
- 증상: 요청이 안 끝남, orchestrator 로그 멈춤.
- 1순위 의심: **ctx-PP → gen-TP 조합**(예 `CTX_PP=2 GEN_TP=2`). 알려진 hang(#14020).
- 조치: `SETUP_LOG.md §Phase 0 지원 매트릭스`에 해당 셀 FAIL 기록 → 그 조합 제외, 또는 1.3.0rc 핀 검토.
- 확인: `LOG_LEVEL=debug` + `UCX_LOG_LEVEL=debug`로 KV전송 단계에서 멈추는지 추적.

### 측정 전 체크 (디버그 OFF 확인)
- [ ] `LOG_LEVEL` 미설정(=info)
- [ ] `TLLM_LOG_LEVEL` / `UCX_LOG_LEVEL` unset
- [ ] `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP` unset (overlap = 정상 동작)
- [ ] `enable_autotuner` 끄지 않음(기본 on — 끄면 성능 저하)
- [ ] perf 수집은 ON 유지 OK (저오버헤드, 측정 후 1회 폴링)

---

## 4. 트러블슈팅 (코드 검증으로 미리 막은 함정)
- **워커 기동 즉시 ValidationError**: extra YAML 키 오타. `enable_block_reuse`/`free_gpu_memory_fraction`는 **반드시 `kv_cache_config:` 하위** (StrictBaseModel extra=forbid).
- **`--model`/`--dtype`/`--served-model-name` 에러**: 그런 CLI 플래그 **없음**. 모델은 positional, dtype은 extra YAML.
- **orchestrator가 포트 무시**: orchestrator host/port는 CLI 불가 → disagg YAML의 `hostname`/`port`에서 읽음 (launch_trtllm.sh가 자동 생성).
- **proxy `/metrics` 404**: orchestrator엔 `/metrics` 없음 → 워커(:8001 등) `/metrics` 또는 proxy `/perf_metrics` 사용.
- **요청이 EOS에서 조기 종료**: `ignore_eos`+`max_tokens`로 길이 강제 (v1.2.1 지원 확인). 필요시 payload에 `min_tokens` 추가.
- **hang (무응답 300s+)**: ctx-PP→gen-TP 조합(#14020) 의심 → `SETUP_LOG.md §Phase 0 지원 매트릭스`에 FAIL 기록, 조합 제외 or 1.3.0rc 검토.

---

## 5. 산출물 & 디버깅
**측정 산출물** (`$EXP_LOG_DIR/<config>/`):
- `bench_p{..}.json` — **공식 benchmark_serving 결과** (TTFT/TPOT/ITL/E2EL/throughput + per-request ITL) ★
- `perf_p{..}.json` — `/perf_metrics` 스냅샷(KV전송시간·블록재사용) ★
- `prom_p{..}.json` — measured 윈도우 전/후 per-side 카운터 스냅샷(prefill/decode RPS 산출용) ★
- `metadata.json` — ctx/gen TP·PP·xPyD·placement·cache_backend
- 시스템: `nvidia_smi.csv`·`ifstat.csv`·`dcgm.log` (1Hz/2s), `s3_sync.log`, `clock_baseline_*`

**분석**: `analyze.py --plot` → **공식 집계** `ttft_p50/p99`·`tpot_p50/p99`·`itl_p99`·`e2el_p99`·`out_tok/s`·`$/Mtok` + `kv_p50/p99`(KV전송시간) **+ `pf_rps`/`dc_rps`/`pf_tps`/`dc_tps`(per-side)** 표·플롯. (kv·per-side는 perf/prom 있을 때만, 없으면 'n/a')

**디버깅 / 로그구조 / KV·perf 읽는 법 / hang 진단 / 디버그 토글 켜고 끄기 → `README.md §디버깅`(위).**
> ⚠️ 측정 런에선 디버그 로그 OFF(`LOG_LEVEL` 미설정). 디버그 로그 = I/O 노이즈 → 변인 오염.
