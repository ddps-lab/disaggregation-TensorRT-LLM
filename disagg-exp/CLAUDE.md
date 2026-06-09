# disagg-exp — TensorRT-LLM PD Disaggregation 실험 (프로젝트-로컬 단일 진실원)

> 이 파일은 이 실험 작업의 **single source of truth**다 (research-rigor #12).
> 글로벌 `~/.claude/CLAUDE.md`가 아니라 **이 repo 안**에 산다. 결정과 그 근거, 핀, 변인통제 규칙을 여기 기록한다.
> 베이스: `NVIDIA/TensorRT-LLM` fork → `ddps-lab/disaggregation-TensorRT-LLM`, 브랜치 `disagg-exp/trtllm-v1.2.1` (태그 `v1.2.1` 기준).

## ⚠️ 작업 방침 (반드시 지킬 것)
- **Claude는 실험 코드를 작성하지 않는다.** 설계·런북·검증·문서만. `launch_trtllm.sh`/YAML/`sweep.py` 수정 등 구현은 **사용자가 직접** 이 fork에서 한다.
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
| CUDA Graph | decode `cuda_graph_config` ON |
| KV 메모리 비율 | `kv_cache_config.free_gpu_memory_fraction: 0.85` (전 config 동일) |
| 결정론 | temp=0, top_p=1, `PYTHONHASHSEED` 고정 |
| 스케줄러 | context 서버 `disable_overlap_scheduler: True` (disagg 표준) |

### 클라이언트측 (`sweep.py` 계승, OpenAI 호환이라 수정 최소)
- token-id prompt(list[int])로 `prefill_len` 고정 + `max_tokens=decode_len`·`ignore_eos=true`로 `decode_len` 강제.
- `stream=true`로 **클라이언트에서 TTFT/E2E 직접 측정** (서버 메트릭은 prefill→전송→decode 전체를 못 봄).
- Poisson 도착, 동시연결 무제한.
- **2-phase**: warmup(10~50) → 건강검진(`fail_rate>0.30` or `ttft_p99>300s` → measured 스킵+`.failed`) → measured **300**(p99 신뢰).

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

## 파일 맵
- **계승(거의 그대로)**: `sweep.py`(orchestrator :8000 조준), `analyze.py`(COST_PER_HR를 g5/g6/g6e 단가로), `setup.sh`(설치만 TRT-LLM 컨테이너로). ← 원본은 `../vllm-disaggregation/disagg-exp/`.
- **신규(사용자 작성)**: `launch_trtllm.sh`(config+role별 trtllm-serve+YAML 생성), ctx/gen/disagg YAML, `trtllm_support_matrix.md`.
- **대체됨**: `launch_configs.sh`→`launch_trtllm.sh`, `disagg_proxy_server.py`→`trtllm-serve disaggregated`, `instrumented_connector.py`→(KV전송시간은 orchestrator 로그/`/metrics`).
- **문서**: `EXPERIMENT_PLAN.md`(전체 설계·런북), 이 `CLAUDE.md`(단일 진실원).

## 관련 조사 기록 (글로벌 메모리)
- `pd-disagg-parallelism-framework-verdict` — 프레임워크 비교(왜 TRT-LLM, vLLM/SGLang 배제).
- `trtllm-disagg-decision` — 채택 결정 + 제약.
- `disagg-exp-experiment-overview` — vLLM 1세대 → TRT-LLM 2세대 전체 개요.
