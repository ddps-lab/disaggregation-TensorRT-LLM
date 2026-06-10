# disagg-exp — TensorRT-LLM PD Disaggregation 실험 (프로젝트-로컬 단일 진실원)

> 이 파일은 이 실험 작업의 **single source of truth**다 (research-rigor #12).
> 글로벌 `~/.claude/CLAUDE.md`가 아니라 **이 repo 안**에 산다. 결정과 그 근거, 핀, 변인통제 규칙을 여기 기록한다.
> 베이스: `NVIDIA/TensorRT-LLM` fork → `ddps-lab/disaggregation-TensorRT-LLM`, 브랜치 `disagg-exp/trtllm-v1.2.1` (태그 `v1.2.1` 기준).

## ⚠️ 작업 방침 (반드시 지킬 것)
- **코드 작성 흐름 (2026-06-10~)**: Claude가 작성. ① v1.2.1 소스 정밀 분석으로 CLI/스키마/키 확정 → ② 코드 작성 → ③ 소스 대조 검증(디버깅 최소화) → ④ 원격 GPU에서 테스트. *추측으로 짜지 말 것 — 코드에서 확인한 사실 위에서만.*
- **공식 코드·방법만 사용** (논문 재현성). 커스텀 해킹 금지. TRT-LLM 공식 `examples/disaggregated/` 패턴을 출발점으로.
- **모든 런타임(서버 기동/추론/벤치)은 원격 EC2에서 SSH로.** 로컬(Mac)은 파일 편집·git 전용. (aws skill 규칙)

## 목표 (Objective — 변수/통제/메트릭)
- **질문**: PD 분리에서 prefill·decode의 병렬화(TP·PP)를 **대칭/비대칭**으로, 토폴로지를 **xPyD(1P1D→1P3D→D확장)**로 바꿀 때 비용대비 성능이 어떻게 변하나.
- **독립변수**: (P의 TP,PP) × (D의 TP,PP) × xPyD(P·D 인스턴스 수) × placement(inter/intra-node).
- **통제변수**: 아래 "변인통제" 표 전부 고정.
- **메트릭**: TTFT p50/p99, TPOT=(e2e−ttft)/(completion−1) p50/p99, throughput 2종(service/arrival window), achieved_rate(saturation), $/Mtok. (정의·계산은 기존 `analyze.py` 그대로 계승)
- **DP는 축이 아님**: dense Qwen3-4B에서 독립 DP 모델-병렬 축 없음. "DP"=attention-DP(MoE/MLA 전용) 또는 xPyD의 D 복제수. 실제 모델-병렬 축 = **(TP, PP)** 둘뿐.

## 확정 스택 + 핀 (Pin — research-rigor #10)
| 항목 | 값 | 검증 |
|---|---|---|
| 프레임워크 | TensorRT-LLM **PyTorch 백엔드** | — |
| 소스 | tag `v1.2.1` (`tensorrt_llm/version.py: __version__="1.2.1"`) | ✅ 코드 확인 |
| 런타임 컨테이너 | **`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`** (통째 핀, NIXL ABI 커플링) | ✅ ref 패턴 확인 |
| 모델 | **Qwen3-4B (dense GQA, non-MLA)** | ✅ `tensorrt_llm/_torch/models/modeling_qwen3.py` 존재 |
| GPU | **g5(A10G SM86)·g6(L4 SM89)·g6e(L40S SM89)** — 전부 SM80+ OK | ✅ |
| dtype | **BF16 통일** (A10G/L4/L40S 전부 native) | — |
| disagg 진입점 | `trtllm-serve disaggregated -c <yaml>` (`tensorrt_llm/commands/serve.py:646`) | ✅ 코드 확인 |
| attention 백엔드 | **TRTLLM** (FlashInfer 금지 — 비대칭 TP 오출력 #6507) | — |
| KV 전송 | `cache_transceiver_config.backend: UCX`, Ethernet 노드 `UCX_TLS=tcp,cuda_copy,sm,self` | ✅ cache_transceiver 코드 존재 |

## Dealbreaker / 알려진 함정 (Front-loaded bad news — #2, #9)
1. **T4(g4dn) 완전 탈락**: disagg 버전엔 T4 지원 없음(하드 abort #2760). → g5/g6/g6e만. (사용자가 T4 불필요 확정 → 무력화됨)
2. **ctx-PP → gen-TP hang (#14020)**: "context pipeline parallelism + generation tensor parallelism" 조합 행. 대표 위험 config = ctx PP2×TP2 → gen PP1×TP4. → **Phase 0에서 300s hang_detector로 반드시 실측.** 막히면 1.3.0rc로 핀 올림 or 해당 조합 제외(사용자와 상의).
3. **무RDMA(AWS xlarge/12xlarge)**: EFA 거의 없음 → inter-node KV전송 = TCP(ENA) → TTFT 100배+ 느림. **inter-node는 구조/correctness 비교용, 깨끗한 성능 결론은 intra-node(PCIe) 중심.**
4. **NVLink 없음(g5/g6/g6e 전부 PCIe)**: intra-node KV전송도 PCIe. cuda_ipc는 Nitro에서 막힐 수 있음 → UCX shm/cuda_copy로.
5. **disagg = EXPERIMENTAL**: 컨테이너 통째 핀 필수(버전 드리프트 시 NIXL ABI 깨짐).
6. **비대칭 TP는 견고, 비대칭 PP는 미보증**: 먼저 비대칭 TP 검증 → 그 다음 비대칭 PP.

## 변인통제 (Hold everything fixed but the variable — #7)
### 서버측 (TRT-LLM YAML/flag) — vLLM에서 매핑
| 통제 의도 | TRT-LLM 설정 |
|---|---|
| prefix/KV 재사용 차단 | `kv_cache_config.enable_block_reuse: false` (PP hang 회피에도 필요) |
| prefill 한 번에(compute-bound) | context 서버 `enable_chunked_prefill: false` |
| dtype 통일 | `--dtype bfloat16` (전 config 동일) |
| 큐 병목 제거 | `max_batch_size` / `max_num_tokens` 상향, 전 config 동일 |
| CUDA Graph | **OFF** — 양쪽 `cuda_graph_config: null` (사용자 결정 2026-06: 균일 eager → 변인통제 + graph-capture cold-start 제거) |
| KV 메모리 비율 | `kv_cache_config.free_gpu_memory_fraction: 0.85` (전 config 동일) |
| 결정론 | temp=0, top_p=1, `PYTHONHASHSEED` 고정 |
| 스케줄러 | context 서버 `disable_overlap_scheduler: True` (disagg 표준) |
| 디버그 로그 | 측정 런 **OFF**(info). 디버그는 smoke에서만(`LOG_LEVEL`/`TLLM_LOG_LEVEL`/`UCX_LOG_LEVEL`). → `DEBUGGING.md` |
| perf 수집 | disagg YAML `perf_metrics_max_requests:1000` + 워커 `return_perf_metrics:true` → `/perf_metrics`, 측정 후 1회 폴링(저오버헤드, ON 유지) |
| autotuner | `enable_autotuner` 기본 ON 유지(끄면 성능 저하). 서버 graph/autotuner는 기동 시 자동 웜업 |

### 클라이언트측 (`sweep.py` 계승, OpenAI 호환이라 수정 최소)
- token-id prompt(list[int])로 `prefill_len` 고정 + `max_tokens=decode_len`·`ignore_eos=true`로 `decode_len` 강제.
- `stream=true`로 **클라이언트에서 TTFT/E2E 직접 측정** (서버 메트릭은 prefill→전송→decode 전체를 못 봄).
- Poisson 도착, 동시연결 무제한.
- **2-phase**: warmup **20**(disagg UCX 첫 연결·스케줄러 ramp 흡수; 서버측 graph/autotuner는 기동 시 자동) → measured **300**(p99 신뢰). **abort/에러율 게이트 제거**(느리거나 실패하는 config도 측정에 남김 — 분석은 status=success 필터). **클라 요청 타임아웃 없음**(서버 `-r` 1800s가 백스톱). smoke 땐 `SWEEP_WARMUP_N=3`. `/health` OK ≠ UCX ready 주의.

### 인프라 (`setup.sh` 계승)
- 좀비 청소(idempotent), **chrony 시계동기**(inter-node 필수), 수집기 nvidia-smi dmon(1Hz)·ifstat(1Hz, NIC=KV전송 관측)·DCGM, S3 자동 sync, TP rank-0만 로깅.

## disagg config 스키마 (v1.2.1 실측 — `examples/disaggregated/disagg_config.yaml`)
```yaml
hostname: localhost
port: 8000
model: <Qwen3-4B 경로>
backend: "pytorch"
disable_overlap_scheduler: True
context_servers:
  num_instances: 1            # = xPyD의 P 개수
  tensor_parallel_size: 1     # P의 TP  ← 비대칭축
  pipeline_parallel_size: 1   # P의 PP  ← 비대칭축
  kv_cache_config: { free_gpu_memory_fraction: 0.85, enable_block_reuse: false }
  cache_transceiver_config: { backend: "UCX" }
  urls: ["localhost:8001"]
generation_servers:
  num_instances: 1            # ★ = xPyD의 D 개수 (1P3D → 3)
  tensor_parallel_size: 1     # D의 TP  ← 비대칭축
  pipeline_parallel_size: 1   # D의 PP  ← 비대칭축
  cache_transceiver_config: { backend: "UCX" }
  urls: ["localhost:8002"]
```
- **`generation_servers.num_instances` = D 개수** (xPyD 핵심 노브). context/generation 각각 독립 TP·PP → 대칭/비대칭 자유.
- 참고 출발점: `examples/configs/curated/qwen3-disagg-prefill.yaml` (단, 이건 Qwen3 **MoE**용 — `enable_attention_dp`/`moe_expert_parallel_size` 포함. Qwen3-4B dense엔 attention_dp 무관, 구조 템플릿은 base `disagg_config.yaml` 사용).

## Phase 0 게이트 (Spike before scale — #8) — multi-GPU 1대(g6e.12xlarge 등)
대량 스윕 전 "무엇이 안 깨지고 도는가"를 실측 매트릭스로 확정:
1. 단일(비분리) sanity: `trtllm-serve <Qwen3-4B> --backend pytorch --tp_size 2` → 로드+추론.
2. 최소 disagg(1 ctx + 1 gen + orchestrator), intra-node, UCX.
3. 조합 매트릭스: 대칭/비대칭 TP(견고) → 대칭/비대칭 PP(위험). 각 셀 = 기동/KV전송/**출력정확성**(비분리 동일 seed 비교).
4. ⚠️ ctx-PP→gen-TP는 #14020 hang 감시.
5. **게이트**: 통과 조합으로만 Phase 1 매트릭스 구성 → `disagg-exp/trtllm_support_matrix.md`에 기록.

## 진행 시퀀스 (사용자 명시)
1. inter-node 1P1D (베이스라인) → 2. 1P3D, decode 병렬화 다양화(대칭→비대칭), inter → 3. intra-node 가정으로 동일 스윕 → 4. D 확장.

## 파일 맵 (코드 작성·검증 완료 2026-06-10)
- **이식(vLLM 계승, 소폭 수정)**: `sweep.py`(MODEL_NAME→Qwen3-4B, metadata 토폴로지, **+perf_metrics 폴링·SWEEP_PD_PAIRS·WARMUP 20**), `analyze.py`(COST_PER_HR g5/g6/g6e, T1~T4, **+KV전송시간 분석**), `setup.sh`(컨테이너 모델, 수집기 보존). ← 원본 `../vllm-disaggregation/disagg-exp/`.
- **도구 추가(2026-06-10, 소스 검증)**: KV전송 계측(`/perf_metrics`→`perf_*.json`→analyze KV컬럼), 디버그 토글(`launch_trtllm.sh` `LOG_LEVEL`+env 인라인주석, `DEBUGGING.md`), smoke(`SWEEP_PD_PAIRS`+적은요청), 웜업(20, 서버/클라 구분), **CUDA graph OFF**. vLLM의 `instrumented_connector.py`는 TRT-LLM 네이티브 `/perf_metrics`로 대체(더 풍부).
- **신규(Claude 작성, v1.2.1 소스 검증)**: `launch_trtllm.sh`(role별 trtllm-serve + disagg YAML 런타임 생성), `ctx_extra_llm_api_options.yaml`·`gen_extra_llm_api_options.yaml`(워커 변인통제), `disagg_config.yaml`(1P1D 정적 템플릿), `trtllm_support_matrix.md`(Phase0 게이트).
- **per-side 측정(2026-06-10, 9-에이전트 소스 전수조사+적대적 검증)**: `prom_scrape.py`(orchestrator `/prometheus/metrics`의 `ctx_/gen_completed_requests_total`를 measured 윈도우 전/후 차분=prefill/decode RPS) → sweep `prom_*.json` → analyze `pf_rps/dc_rps/pf_tps/dc_tps`. **per-side TPS는 공식 토큰 카운터 부재**(collector.py 토큰카운터 0)로 `RPS×고정길이` 파생. **전체 TTFT/TPOT/throughput/latency는 sweep이 이미 측정**(공식 `benchmark_serving`과 정의 동일 — 부하코어 교체 불필요, 선택적 교차검증만). 상세·근거=`병렬화-KV전송-측정.md §6/§7`.
- **대체됨**: `launch_configs.sh`→`launch_trtllm.sh`, `disagg_proxy_server.py`→`trtllm-serve disaggregated`, `instrumented_connector.py`→(KV전송시간은 orchestrator `/perf_metrics`).
- **코드 검증**: 적대적 워크플로우로 v1.2.1 소스 대조 → blocker 0. 빈배열 가드(bash<4.4 이식성) 등 minor 픽스 반영. 로컬 정적검사(bash -n / py_compile / yaml) 통과. **런타임 검증은 원격 GPU Phase 0에서.**
- **문서 (8종, 역할 분리 — "한 사실은 한 문서에", 중복 금지)**. 실험 중 무엇을 열지 한눈에:

  | 문서 | 한 줄 역할 | 언제 여나 | 정본 항목(여기서만 풀버전) |
  |---|---|---|---|
  | **`CLAUDE.md`** (이 파일) | 결정·핀·**변인통제·dealbreaker·disagg 스키마·프레임워크판정** | 항상(자동로드) | ★ 이 4가지의 **정본**. 다른 문서는 링크만 |
  | **`README.md`** | **실행 가이드** (컨테이너→smoke→single/inter-node, env표, 트러블슈팅) | 돌릴 때 | 실행 절차·env 레퍼런스 |
  | **`EXPERIMENT_PLAN.md`** | **무엇을/왜/어떤 순서로** (실험축·Phase 0~3 런북·워크로드 그리드·vLLM→TRT 매핑) | 설계 확인 | Phase 0~3·그리드·매핑표 |
  | **`DEBUGGING.md`** | 디버그 토글·로그구조·perf/KV전송 읽기·hang(#14020) 진단·측정전 OFF 체크 | 디버깅할 때 | 로그 디렉토리 구조 |
  | **`LEARNING_NOTES.md`** | 개념 공부 노트(fork/PD분리/TP·PP/메트릭/웜업) — 교과서 수준, 살아있는 문서 | 공부할 때 | 개념 설명 |
  | **`병렬화-KV전송-측정.md`** | 병렬화×cache_transceiver×**per-side 측정**(prefill/decode RPS·TPS) 심화 조사, 소스레퍼런스 | 측정 설계/조사 | per-side 측정법·KV 백엔드·지원매트릭스 근거 |
  | **`trtllm_support_matrix.md`** | Phase 0 게이트 **결과표**(어떤 TP·PP 조합이 실제로 도나 — 실측 채움) | Phase 0 실측 | 게이트 실측 결과(산출물) |
  | **`SETUP_LOG.md`** | 시간순 **작업 로그 + provenance** (어디서 무엇을 했나) | 회고/methods 작성 | 작업 이력 |

## 프레임워크 선택 근거 + 배경 (구 글로벌 메모리 통합 — 이 문서가 단일 진실원)

### 프레임워크 판정 (왜 TRT-LLM, 나머지 배제 — 공식 코드/이슈/PR로 검증 2026-06)
- **vLLM**: 비대칭 TP는 NixlConnector만 가능. **PP-in-PD 불가**(KV 전송 프로토콜에 PP 필드 없음, #40674). LMCache는 둘 다 ❌. → **배제.** (← 이 실험이 vLLM에서 넘어온 직접 이유)
- **SGLang**: PD KV 전송이 PP-aware(`base/conn.py`의 `KVArgs`에 `pp_rank`/`prefill_start_layer`/`prefill_end_layer`)하지만, **임의 비대칭 PP는 하드 assert로 차단** — `common/conn.py`의 `_resolve_rank_mapping`에 `assert pp_size == info.pp_size or pp_size == 1` ("Decode pp size should be equal to prefill pp size or 1"). 즉 **"prefill PP=N → decode PP=1" gather만** 가능. **대칭 PP-in-PD조차 크래시/출력손상 버그**(#15571 `PPMissingLayer.quant_method`, #16246; 수정 #19804는 #21189로 리버트). 비대칭 TP는 MLA 견고/비-MLA는 부하버그 #15674 OPEN. 무RDMA는 `mooncake_tcp` 또는 NIXL/UCX-TCP. ⚠️ **`--disaggregation-decode-tp` 플래그는 없음** — 비대칭 TP는 각 서버에 다른 `--tp-size`. → **배제.**
- **TensorRT-LLM**: **임의 비대칭 PP+TP를 1급으로 지원하는 유일한 유지보수 프로덕션 엔진**(context/generation 서버별 독립 tp/pp, UCX-over-TCP로 무EFA 가능). C++ `cacheFormatter.cpp::inquireSupport`는 PP 조합이 아니라 KV 동질성(dtype·head수·layer수·non-MLA·beam=1)만 검사 → dense GQA 충족. → **채택.**
- **Dynamo / llm-d**: vLLM 백엔드의 PP 한계를 그대로 상속 → 비대칭 PP 불가.
- **DistServe**(연구용): 비대칭 TP+PP는 깔끔하나 **per-phase DP 없음** + **CUDA IPC라 멀티노드 KV 전송 불가** → 단일노드 소형모델 연구용.

### DP 교차사실 (dense 모델 한정 — 중요)
"DP 안 됨"은 TRT-LLM 한정이 아니라 **dense GQA에선 어떤 엔진에서도 DP가 독립 모델-병렬 축이 아님**. DP의 3가지 의미: (1) 독립 샤딩 knob — TRT-LLM엔 아예 없음, vLLM/SGLang `--data-parallel-size`는 dense를 통째 복제, (2) attention-DP — MLA/MoE 전용(GQA 무의미), (3) 복제본 수/xPyD — 이것만 해당(리소스 배분). → **실험의 진짜 독립 축 = TP·PP 둘. DP는 인스턴스 수(xPyD)로만.** DP/EP를 진짜 축으로 보려면 MLA/MoE 모델(DeepSeek) 필요.

### T4(SM75) 교차사실 (중요)
T4 미지원은 **TRT-LLM·Dynamo만의 문제(최소 SM80)**. vLLM=T4 지원(SM75), SGLang=되나 과도기(prebuilt 휠에서 sm75 제거 #9207 → 소스빌드/구버전 핀 + Triton 3.2.0 다운). → 우리는 **T4 불필요 확정** → g5/g6/g6e(전부 SM80+).

### 1세대(vLLM) 배경 — 방법론의 출처
vLLM v0.21 fork + LMCache/NIXL, **Llama-3.1-8B**, AWS. 7개 config — monolithic(A1 TP2PP2 / A2 TP4PP1 / A3 TP1PP4, 4×T4) vs single big GPU(B, L40S) vs same-node PD(C, 4×T4 shm) vs cross-node PD(D, 2×L4 TCP). 측정 TTFT/TPOT/$per-Mtoken. vLLM이 **PD에서 PP 불가**라 2세대(TRT-LLM+Qwen3-4B)로 이전하며 sweep.py·analyze.py·setup.sh·2-phase·변인통제 **계승**. 원본 = `../vllm-disaggregation/disagg-exp/`.

### 전략 메모
Qwen3-4B(및 Llama-8B)는 GPU 1장에 올라가므로 **PP는 "크로스노드 메모리 분산" 용도** — 그게 바로 대부분 프레임워크에서 막힌 케이스(= 이 연구가 파고드는 지점). PP를 축으로 보는 연구적 의미가 약하면 더 큰 모델도 고려 가능.
