# disagg-exp — TensorRT-LLM PD Disaggregation 실험 (프로젝트-로컬 단일 진실원)

> 이 파일은 이 실험 작업의 **single source of truth**다 (research-rigor #12). 글로벌 `~/.claude/CLAUDE.md`가 아니라 **이 repo 안**에 산다.
> 베이스: `NVIDIA/TensorRT-LLM` fork → `ddps-lab/disaggregation-TensorRT-LLM`, 브랜치 `disagg-exp/trtllm-v1.2.1` (태그 `v1.2.1` 기준), 런타임 컨테이너 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`.

## ⚠️ 작업 방침 (반드시 지킬 것)
- **코드 작성 흐름 (2026-06-10~)**: Claude가 작성. ① v1.2.1 소스 정밀 분석으로 CLI/스키마/키 확정 → ② 코드 작성 → ③ 소스 대조 검증(디버깅 최소화) → ④ 원격 GPU에서 테스트. *추측으로 짜지 말 것 — 코드에서 확인한 사실 위에서만.*
- **공식 코드·방법만 사용** (논문 재현성). 커스텀 해킹 금지. TRT-LLM 공식 `examples/disaggregated/` 패턴을 출발점으로.
- **모든 런타임(서버 기동/추론/벤치)은 원격 EC2에서 SSH로.** 로컬(Mac)은 파일 편집·git 전용. (aws skill 규칙)

## 목표 (Objective — 변수/통제/메트릭)
- **질문**: PD 분리에서 prefill·decode의 병렬화(TP·PP)를 **대칭/비대칭**으로, 토폴로지를 **xPyD(1P1D→1P3D→D확장)**로 바꿀 때 비용대비 성능이 어떻게 변하나.
- **독립변수**: (P의 TP,PP) × (D의 TP,PP) × xPyD(P·D 인스턴스 수) × placement(inter/intra-node).
- **통제변수**: 아래 "변인통제" 표 전부 고정.
- **메트릭**: TTFT p50/p99, TPOT=(e2e−ttft)/(completion−1) p50/p99, throughput 2종(service/arrival window), achieved_rate(saturation), **per-side prefill/decode RPS·TPS**. (정의·계산 `analyze.py` 계승)
  - **비용은 analyze에서 빼고**(2026-06-24 결정) writeup에서 외부 계산: `$/Mtok = (인스턴스수 × $/hr) / output_throughput`. throughput은 측정·기록되므로 그 시점 단가로 언제든 재계산(코드에 단가 박으면 region/spot/시점에 stale). analyze는 측정치만 출력.
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
| KV 전송 | `cache_transceiver_config.backend: NIXL`(=DEFAULT, 공식 벤치 일치). no-EFA에선 내부 transport=UCX → `UCX_TLS=tcp,cuda_copy,sm,self`. 폴백=UCX | ✅ cache_transceiver 코드 존재 |

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
| 디버그 로그 | 측정 런 **OFF**(info). 디버그는 smoke에서만(`LOG_LEVEL`/`TLLM_LOG_LEVEL`/`UCX_LOG_LEVEL`). → `README.md §디버깅` |
| perf 수집 | disagg YAML `perf_metrics_max_requests:1000` + 워커 `return_perf_metrics:true` → `/perf_metrics`, 측정 후 1회 폴링(저오버헤드, ON 유지) |
| autotuner | `enable_autotuner` 기본 ON 유지(끄면 성능 저하). 서버 graph/autotuner는 기동 시 자동 웜업 |

### 클라이언트측 (부하코어 = 공식 `benchmark_serving`, sweep는 오케스트레이션)
- **부하·측정 = 공식 `benchmark_serving` 서브프로세스**(sweep가 `build_bench_args`로 호출). 손수 짠 aiohttp 제거. **benchmark_serving이 부하 코어다.**
- ISL/OSL 고정: `--random-ids --tokenize-on-client --random-range-ratio 0.0`(token-id `prompt_token_ids`로 정확) + `--ignore-eos --random-output-len`로 OSL.
- TTFT/TPOT/ITL/E2EL = 공식 클라측 측정(SSE) → `bench_<point>.json`. **`--percentile-metrics ttft,tpot,itl,e2el` 필수**(아니면 E2EL 누락).
- Poisson 도착 `--request-rate R --burstiness 1.0`, 동시연결 무제한(`--max-concurrency` 미설정).
- **2-phase**: warmup **20**(measured 전 소량 `--non-streaming` 호출 — benchmark_serving 내장 warmup 없음, disagg UCX cold-start 흡수) → measured **300**. smoke 땐 `SWEEP_WARMUP_N=3`. abort 게이트 없음(실패 config도 `.failed` 마커만, 분석은 result.json 유무로). `/health` OK ≠ UCX ready 주의.

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
  cache_transceiver_config: { backend: "NIXL" }   # =DEFAULT. no-EFA→내부 UCX-TCP. 폴백 UCX
  urls: ["localhost:8001"]
generation_servers:
  num_instances: 1            # ★ = xPyD의 D 개수 (1P3D → 3)
  tensor_parallel_size: 1     # D의 TP  ← 비대칭축
  pipeline_parallel_size: 1   # D의 PP  ← 비대칭축
  cache_transceiver_config: { backend: "NIXL" }   # =DEFAULT. no-EFA→내부 UCX-TCP. 폴백 UCX
  urls: ["localhost:8002"]
```
- **`generation_servers.num_instances` = D 개수** (xPyD 핵심 노브). context/generation 각각 독립 TP·PP → 대칭/비대칭 자유.
- 이 스키마는 참조용. 실제 disagg config는 `launch_trtllm.sh`가 `write_disagg_yaml()`로 런별 런타임 생성(`$LOG_DIR/disagg_<LABEL>.yaml`). 참고 출발점: `examples/configs/curated/qwen3-disagg-prefill.yaml` (단, 이건 Qwen3 **MoE**용 — `enable_attention_dp`/`moe_expert_parallel_size` 포함. Qwen3-4B dense엔 attention_dp 무관).

## 문서 인덱스
문서는 8종 → **4종**으로 통합(2026-06-23). 한 사실은 한 문서에:

| 무엇을 보나 | 문서 | 비고 |
|---|---|---|
| 실행·실험 순서·변인통제·디버깅 | **`README.md`** | 실행 가이드 + §실험 순서 / 로드맵 · §변인 통제 · §디버깅 (구 EXPERIMENT_PLAN·DEBUGGING 흡수) |
| 개념·프레임워크 판정·병렬화 조사 | **`LEARNING_NOTES.md`** | 공부 노트 + §병렬화·KV전송·per-side(구 `병렬화-KV전송-측정.md`) + §프레임워크 판정·배경(구 CLAUDE 블록) |
| 작업 기록·Phase 0 결과 | **`SETUP_LOG.md`** | 시간순 작업 로그 + provenance + §Phase 0 지원 매트릭스(구 `trtllm_support_matrix.md`) |
| 결정·핀·변인통제·dealbreaker·스키마 | **`CLAUDE.md`** (이 파일) | 자동 로드되는 thin 계약. 정본 |

> 구 8종(CLAUDE·README·EXPERIMENT_PLAN·DEBUGGING·LEARNING_NOTES·병렬화-KV전송-측정·trtllm_support_matrix·SETUP_LOG) → 현 4종. 정적 `disagg_config.yaml`은 삭제(런타임 생성으로 대체). 변인통제 정본은 이 파일에 유지(자동 로드 계약).
