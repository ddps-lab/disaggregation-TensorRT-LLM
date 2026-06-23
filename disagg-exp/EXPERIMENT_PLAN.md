# 실험 설계: TensorRT-LLM PD Disaggregation — Qwen3-4B, xPyD + 대칭/비대칭 TP·PP (g5/g6/g6e)

> **역할**: *무엇을/왜/어떤 순서로* 측정하나 = 설계·런북. **핀·변인통제값·dealbreaker·disagg 스키마·프레임워크판정의 정본은 `CLAUDE.md`** — 여기선 중복 서술하지 않고 링크한다. 이 문서 고유 = 실험축·Phase 0~3·워크로드 그리드·vLLM→TRT 매핑.

## Context (왜 / 무엇을)

목표: Prefill/Decode 분리(PD disaggregation)에서 **prefill·decode의 병렬화(TP·PP)를 대칭/비대칭으로 바꿔가며**, 그리고 **xPyD 토폴로지(1P1D → 1P3D → 확장)**로, **inter-node·intra-node** 둘 다에서 성능을 측정한다. 모델은 **Qwen3-4B (dense, GQA, non-MLA)**.

프레임워크 선택 결론 — **TensorRT-LLM (PyTorch 백엔드)**. 근거(선행 조사 검증 완료):
- PD 분리에서 **임의 비대칭 PP+TP를 parse-time assert 없이 1급 지원하는 유일한 프로덕션 엔진**. vLLM=PD에서 PP 전면 불가(#40674), SGLang=비대칭 PP가 prefill-PP→decode-PP=1로만 제한+대칭 PP도 버그(#15571). 자세한 비교: **`CLAUDE.md` §프레임워크 판정** (구 메모리 통합본).
- TRT-LLM의 유일한 걸림돌이던 **T4 미지원이 무효화됨**(사용자가 T4 안 써도 됨). 이제 쓸 GPU(g5=A10G SM86, g6=L4 SM89, g6e=L40S SM89)는 **전부 TRT-LLM 호환(최소 SM80 충족)**.
- PyTorch 백엔드라 **config별 엔진 빌드 불필요**(HF 직접 로드) → TP/PP 스윕에 적합. OpenAI 호환 엔드포인트라 **기존 sweep.py 재사용 가능**.

**DP는 축이 아님**: dense Qwen3-4B에서 DP는 독립 모델-병렬 축이 아니라 복제본 수(=xPyD의 D 개수)일 뿐. 따라서 실험의 모델-병렬 축은 **TP·PP 둘**, 그리고 토폴로지 축은 **xPyD(P·D 인스턴스 수)**.

> 방침 (2026-06-10 변경): **코드는 Claude가 TRT-LLM v1.2.1 소스에 맞춰 작성.** 흐름 = ① 소스 정밀 분석으로 정확한 CLI/스키마/키 확정 → ② 그 위에서 코드 작성(launch_trtllm.sh·YAML·sweep/analyze/setup 이식) → ③ 소스 대조 검증으로 코드 정확성 확인(디버깅 최소화) → ④ **원격 GPU에서 Phase 0 스파이크로 실제 동작 테스트.** 모든 런타임은 원격 EC2에서 SSH로(`aws` skill), 로컬은 파일·git.

## 하드웨어 / 전송 현실 (EC2, 검증됨)

| 항목 | 사실 | 함의 |
|---|---|---|
| GPU | g5=A10G(SM86), g6=L4(SM89), g6e=L40S(SM89) | 전부 TRT-LLM OK. dtype: A10G/L4/L40S 모두 FP16/BF16(+FP8 Ada). 공정 비교는 **BF16 또는 FP16 통일** |
| NVLink | g5/g6/g6e **없음**(PCIe만) | intra-node KV전송 = PCIe. cuda_ipc는 Nitro에서 막힐 수 있음 → UCX가 shm/cuda_copy로 처리 |
| EFA | xlarge·12xlarge 대부분 **없음** (최상위 사이즈만) | **inter-node KV전송 = TCP(ENA) → 느림.** inter-node는 구조/correctness용, 깨끗한 성능 결론은 intra-node 위주 |
| 전송 백엔드 | TRT-LLM 기본 UCX, TCP fallback | `cache_transceiver_config.backend: UCX` 고정. Ethernet 노드 `UCX_TLS=tcp,cuda_copy,sm,self` |

## 실험 축 (사용자 진행 순서 반영)

- **토폴로지(xPyD)**: 1P1D → **1P3D** → 확장(D 개수↑). decode가 throughput 병목이라 D 스케일이 핵심.
- **병렬화**: prefill (TP_p, PP_p), decode (TD_d, PP_d) — **대칭(P=D)·비대칭(P≠D)** 둘 다. 특히 "D에 여러 병렬화" = 각 decode 인스턴스의 TP/PP 변형.
- **배치**: inter-node 먼저 → intra-node.
- **측정**: 기존 하네스의 TTFT / TPOT / throughput(2종) / $-per-Mtok (analyze.py 그대로).

진행 시퀀스(사용자 명시):
1. **inter-node 1P1D** (베이스라인)
2. **1P3D, decode 병렬화 다양화, inter-node** (대칭→비대칭)
3. **intra-node 가정으로 동일 스윕**
4. **D 확장**(D3→그 이상)

## 변인 통제 + 스윕 설계 (기존 vLLM `disagg-exp`에서 계승 — vLLM이 PP가 안 돼서 넘어온 것)

이 실험은 `vllm-disaggregation/disagg-exp`의 방법론을 그대로 잇는다. **클라이언트 측(`sweep.py`)·메트릭(`analyze.py`)·2-phase·수집기·시계동기·S3는 프레임워크 무관 → 거의 그대로 재사용**, **서버 측 noise control만 TRT-LLM 플래그로 매핑**한다.

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

### ②~④ 클라이언트·메트릭·인프라 통제 → 정본 **`CLAUDE.md` §변인통제**
중복 서술 대신 정본을 가리킨다. 여기선 이 실험의 **계승 포인트**(무엇을 vLLM에서 그대로 잇나)만:
- **부하·측정(공식 `benchmark_serving`, sweep는 오케스트레이션 — 2026-06-11)**: sweep가 포인트마다 공식 도구를 서브프로세스 호출(`--random-ids --tokenize-on-client --random-range-ratio 0`로 token-id ISL 고정, `--ignore-eos`로 OSL, `--request-rate --burstiness 1.0` Poisson, `--percentile-metrics ttft,tpot,itl,e2el`). warmup=measured 전 소량 `--non-streaming` 호출. 포인트당 `/perf_metrics` 1회 폴링(KV) + per-side 스냅샷. *상세=CLAUDE / 병렬화-KV전송-측정 §7.*
- **메트릭(`analyze.py` 계승)**: TTFT·TPOT·throughput 2종(service/arrival window)·achieved_rate·$/Mtok. *정의·공식=CLAUDE §목표 / `LEARNING_NOTES.md §E`.*
- **인프라(`setup.sh` 계승)**: 좀비청소·chrony(inter-node 타임스탬프 정렬 필수)·수집기(nvidia-smi/ifstat=NIC·DCGM)·S3 sync·TP rank-0만 로깅. *상세=CLAUDE.*

### ⑤ 워크로드 그리드 (계승 + Qwen3-4B/하드웨어 맞춰 조정)
- **PD_PAIRS**(prefill,decode): `(2048,128)` prefill-heavy · `(1024,512)` balanced · `(128,2048)` decode-heavy — 3종 유지.
- **RATES**: `[1.0, 2.0, 4.0]`(저/중/포화). saturation 보려면 상향 가능.
- **max-model-len**: T4용 4096 제약 **해제**(L4 24GB/L40S 48GB) → 더 긴 조합 가능하나 비교 위해 고정값 명시.
- 각 config의 `metadata.json`에 **P/D의 (TP,PP) + xPyD(P·D 수) + placement(inter/intra) + EFA 유무** 기록(기존 `tp_map/pp_map` 패턴 확장) → 전송 변수 분리.

## Phase 0 — TRT-LLM 검증 스파이크 (게이트, multi-GPU 1대)

대량 스윕 전, "무엇이 안 깨지고 도는가"를 실측 매트릭스로 확정. (선행 조사에서 함정 식별됨)

1. **환경**: g6e.12xlarge(4×L40S) 또는 g6.12xlarge(4×L4) 1대. 컨테이너/버전 **통째 핀** — 시작점 **v1.2.1**(+ NIXL 버전 포함). `--backend pytorch`.
2. **단일(비분리) sanity**: `trtllm-serve Qwen3-4B --backend pytorch --tp_size 2`로 Qwen3-4B 로드+추론 확인 (Qwen3 지원 버전 확인).
3. **최소 disagg (1 ctx + 1 gen + orchestrator)**, intra-node, TCP 전송:
   ```bash
   # context
   CUDA_VISIBLE_DEVICES=0,1 trtllm-serve Qwen3-4B --backend pytorch --tp_size 2 --port 8001 \
     --extra_llm_api_options ctx.yaml   # cache_transceiver_config.backend: UCX; disable_overlap_scheduler: True
   # generation
   CUDA_VISIBLE_DEVICES=2,3 trtllm-serve Qwen3-4B --backend pytorch --tp_size 2 --port 8002 \
     --extra_llm_api_options gen.yaml   # enable_block_reuse: false
   # orchestrator (OpenAI 호환 단일 엔드포인트)
   trtllm-serve disaggregated -c disagg.yaml   # context_servers / generation_servers URL 나열
   ```
4. **조합 매트릭스 실측** → `disagg-exp/trtllm_support_matrix.md`:
   - 대칭 TP: P(tp2)·D(tp2) / 비대칭 TP: P(tp1)·D(tp2), P(tp2)·D(tp1)
   - 대칭 PP: P(pp2)·D(pp2) / 비대칭 PP: P(pp2)·D(pp1), P(pp1)·D(pp2)
   - ⚠️ **ctx-PP→gen-TP(예: P pp2×tp1 → D pp1×tp2)는 알려진 hang(#14020)** → 300s hang_detector 감시. 막히면 1.3.0rc로 핀 변경 재시도.
   - 각 셀: 기동 성공 / 첫 요청 KV전송 성공 / **출력 정확성**(비분리 동일 seed와 비교, 과거 비대칭 TP garbled #6507 회귀 점검; **TRTLLM attention 백엔드 사용, FlashInfer 금지**).
5. **게이트**: 통과 조합으로만 Phase 1 매트릭스 구성. PP가 광범위하게 막히면 사용자와 재상의(버전 올림 vs 축 축소).

## Phase 1 — 하네스 (기존 자산 재사용 최대화)

| 기존 (vLLM) | 처리 |
|---|---|
| `disagg-exp/sweep.py` | **오케스트레이션으로 개편** — 부하코어=공식 benchmark_serving 서브프로세스(:8000 조준). 그리드·resume·S3·metadata·warmup·per-side 스냅샷만 담당 |
| `disagg-exp/analyze.py` | 재사용 + config 리스트·`COST_PER_HR`(g5/g6/g6e 단가) 갱신 |
| `disagg-exp/setup.sh` | metric collector(nvidia-smi/ifstat/dcgm)·chrony 재사용, 설치단계만 TRT-LLM 컨테이너로 |
| `launch_configs.sh` | **신규 `launch_trtllm.sh`로 대체** — config+role(context/generation/orchestrator)별 `trtllm-serve` 명령 + YAML 생성. 기존 디스패처 구조 차용 |
| `disagg_proxy_server.py` | **`trtllm-serve disaggregated`(orchestrator)로 대체** |
| `instrumented_connector.py` | 대체 — KV전송시간은 orchestrator `/perf_metrics`(→ `perf_*.json`, analyze가 읽음) |

변인 통제 유지(기존 그대로, 명칭만 매핑): prefix/radix cache off, dtype 통일(BF16/FP16), 2-phase warmup/measured, seed, `enable_block_reuse:false`, **TRTLLM attention 백엔드**.

## Phase 2 — Config 매트릭스 (스파이크 통과분 위에서)

`(xPyD, P:[TP,PP], D:[TP,PP], placement)` 격자. 예시 시작점:
- T1: inter 1P1D, P tp1·D tp1 (대칭 베이스라인)
- T2: inter 1P3D, D 각각 tp1 (복제) → tp2 → pp2 (decode 병렬화 다양화)
- T3: 비대칭 — P tp2 → D tp1 / P tp1 → D tp2
- T4: intra 가정으로 T1~T3 반복 (PCIe 전송)
각 config는 inter/intra와 EFA 유무를 로그에 명시(전송 변수 분리).

## Phase 3 — 검증 (end-to-end)

1. Phase 0 게이트 통과.
2. Smoke: `launch_trtllm.sh <cfg> {context,generation,orchestrator}` → `/health` → `sweep.py --config smoke` 1 point → `bench_*.json` 생성(공식 결과).
3. 비대칭 TP 1 point에서 KV전송/출력정확성 sanity.
4. small sweep → full sweep (기존 `.done/.failed` resume).
5. `analyze.py --plot` — TTFT/TPOT/$Mtok + (xPyD, 병렬화 조합)별 비교.

## 핵심 파일
- 신규: `disagg-exp/launch_trtllm.sh`, `disagg-exp/trtllm_support_matrix.md`, ctx/gen/disagg YAML
- 수정(소폭): `disagg-exp/sweep.py`, `analyze.py`, `setup.sh`, `README.md`
- 대체됨: `launch_configs.sh`, `instrumented_connector.py`, `disagg_proxy_server.py`

## 리스크 / 사용자 상의 (전체 목록·근거 = **`CLAUDE.md` §Dealbreaker**)
Phase 0 결과로 **결정할 것**만 여기 둔다:
- **ctx-PP→gen-TP hang(#14020)**: 핀 버전에서 재현 시 → 1.3.0rc 핀 올림 vs 해당 조합 제외 (Phase 0 실측으로 결정).
- **inter-node TCP**: EFA 없어 KV전송이 TTFT 지배하면 → inter는 구조/correctness 비교용, **성능 결론은 intra-node 중심**.
- **PP가 광범위하게 막히면** → 축 축소 vs 버전 올림, 사용자와 상의.
