# 병렬화 × KV 전송(cache_transceiver) × per-side 측정 — TRT-LLM 판

> **상태: TRT-LLM 적응본.** 이 문서는 vLLM 1세대 조사노트(`../../vllm-disaggregation/disagg-exp/병렬화-커넥터.md`)의
> **방법론·구조만 차용**하고, 내용은 전부 **TensorRT-LLM v1.2.1 소스 기준**으로 다시 썼다 (vLLM 복붙 아님).
> *(파일명 `병렬화-커넥터`→`병렬화-KV전송-측정`: TRT-LLM엔 vLLM 같은 "커넥터" 개념이 없고 cache_transceiver를 쓰므로 이름을 내용에 맞춤.)*
> vLLM 문서가 "어떤 *커넥터*가 어떤 병렬화를 지원하나"였다면, TRT-LLM은 커넥터 플러그인 개념이 없고
> `cache_transceiver`로 KV를 옮기며 **임의 비대칭 PP+TP를 1급 지원**한다 → 표·결론이 근본적으로 다르다.
> 대상: `ddps-lab/disaggregation-TensorRT-LLM` @ `disagg-exp/trtllm-v1.2.1`. file:line은 v1.2.1 기준(드리프트 시 재grep).

---

## 0. 핵심 용어 — cache_transceiver ≠ transport (vLLM의 커넥터≠트랜스포트에 대응)

| | 정체 | 결정 | TRT-LLM 값 |
|---|---|---|---|
| **cache_transceiver** | KV를 누구→누구로 언제 옮길지 + ctx/gen 레이아웃 매핑. `CacheTransceiverConfig` | **호환성**(어떤 KV를 보낼 수 있나) | `backend: DEFAULT\|UCX\|NIXL\|MOONCAKE\|MPI` |
| **transport** | 바이트를 옮기는 배관 | **속도/경로** | UCX TLS (tcp, cuda_copy, sm, cuda_ipc, rdma…) |

**vLLM과의 결정적 차이**: vLLM은 커넥터(P2pNccl/LMCache/NIXL)마다 PP·비대칭TP 지원이 갈렸다. **TRT-LLM은 cache_transceiver 백엔드가 무엇이든 비대칭 PP·TP를 막지 않는다** — 호환성 게이트는 *병렬화 조합*이 아니라 *KV 동질성*(dtype·head수·layer수·non-MLA·beam=1)만 본다(`cacheFormatter.cpp::inquireSupport`). 그래서 dense GQA Qwen3-4B는 비대칭 TP·PP를 1급으로 통과한다. (이게 vLLM에서 TRT-LLM으로 넘어온 직접 이유 — `CLAUDE.md` 프레임워크 판정.)

- 우리 설정: `cache_transceiver_config.backend: UCX`, Ethernet 노드 `UCX_TLS=tcp,cuda_copy,sm,self`.
- `DEFAULT == NIXL` (examples README). 우리는 명시적으로 UCX 고정.
- 근거: `tensorrt_llm/llmapi/llm_args.py:1807-1843` (CacheTransceiverConfig), `examples/disaggregated/README.md`.

---

## 1. KV 전송 백엔드 (vLLM의 "커넥터 종류 factory.py"에 대응)

vLLM처럼 커넥터 클래스를 고르는 게 아니라, **disagg YAML / 워커 extra YAML에 backend 문자열 하나**로 정한다.

| `cache_transceiver_config.backend` | 비고 |
|---|---|
| `UCX` | 우리 기본. UCX TLS로 transport 경로 제어. 컨테이너 사전설치 |
| `NIXL` | `DEFAULT`의 실체. 별도 플러그인 빌드 필요할 수 있음 |
| `MOONCAKE` | 외부 KV store. 본 실험 범위 밖 |
| `MPI` | 레거시. DEPRECATED 경로 |
| `DEFAULT` | =NIXL |

- ctx/gen **양쪽 동일 backend**여야 KV 전송 호환 (우리 ctx/gen extra YAML 둘 다 UCX).
- vLLM의 `--kv-transfer-config` JSON·`kv_producer/kv_consumer`·`kv_port`·`send_type=PUT_ASYNC`·`nccl_num_channels` 같은 P2pNccl 잡설정은 **TRT-LLM엔 없음** (orchestrator가 알아서 ctx→gen 라우팅).

---

## 2. "decode를 어떻게 스케일하나" — 병렬화 3분류 (개념은 공통, KV 전송 관점은 TRT-LLM)

| 방식 | 정체 | KV 전송 관점 | TRT-LLM 표현 |
|---|---|---|---|
| **DP**(복제) | full-model decode 인스턴스 N개, 라우터 분배 | KV 레이아웃 변환 불필요 | `generation_servers.num_instances: N` (xPyD) |
| **TP**(분할) | decode 1인스턴스를 N GPU에 head 분할 | 비대칭 TP면 head 재매핑 | `generation_servers.tensor_parallel_size: N` |
| **PP**(파이프라인) | decode 1인스턴스 layer를 stage 분할 | layer→stage KV 라우팅 | `generation_servers.pipeline_parallel_size: N` |

**dense 모델 주의(중요)**: Qwen3-4B(dense GQA)엔 **독립 DP 모델-병렬 축이 없다.** "DP"는 (1) xPyD 복제수(`num_instances`)이거나 (2) attention-DP(MoE/MLA 전용, GQA 무의미)일 뿐. → 우리 실제 모델-병렬 축 = **TP·PP 둘**, 토폴로지 축 = **xPyD(P·D 인스턴스 수)**. (vLLM 문서의 "DP=복제" 부분만 유효, attention-DP 칸은 dense엔 해당 없음.)

---

## 3. 지원 매트릭스 — TRT-LLM (vLLM과 정반대로 대부분 ✅)

범례: ✅ 1급 지원 / ⚠️ 지원하나 런타임 검증 필요 / ❌

| 병렬화 (P=prefill, D=decode) | vLLM 1세대(참고) | **TRT-LLM v1.2.1** | 근거 |
|---|---|---|---|
| DP / xPyD (복제 N) | ✅ | ✅ | `num_instances` |
| 대칭 TP (P=D) | ✅ | ✅ | 견고 |
| **비대칭 TP** (P tp1 → D tp4) | NixlConnector만 ✅ | **✅** | ctx/gen 독립 `tensor_parallel_size`, head 재매핑 |
| 대칭 PP (P=D) | ❌(WIP) | **⚠️** | 독립 `pipeline_parallel_size` 허용. 런타임 확인 |
| **비대칭 PP** (P pp1 → D pp4) | ❌ 전 커넥터 | **⚠️** | 막는 assert 없음. inquireSupport는 KV 동질성만 검사 |
| ⚠️ **ctx-PP → gen-TP** (P pp2 → D tp2) | — | **⚠️ hang 위험** | **알려진 #14020 hang** |

> 🔑 **핵심 결론 (vLLM과 반대)**:
> - vLLM은 "**PP-disagg를 아무 커넥터도 안 짜서**(WIP #40674) 전부 ❌, 비대칭 TP는 NIXL만"이었다.
> - **TRT-LLM은 ctx/gen이 각각 독립 `tensor_parallel_size`/`pipeline_parallel_size`를 받고, parse-time에 비대칭 PP/TP를 막는 assert가 없다.** 호환성 검사는 KV 동질성(dtype·head·layer·non-MLA·beam=1)만 → dense GQA Qwen3-4B 충족.
> - **단 "코드가 안 막음 ≠ 실제로 안 깨짐"** — 특히 **ctx-PP→gen-TP는 알려진 hang(#14020)**. → 어떤 (TP,PP) 조합이 실제로 도는지는 **Phase 0 스파이크 게이트**(`trtllm_support_matrix.md`)로 실측 확정. 비대칭 TP 먼저(견고) → 비대칭 PP(미보증) 순.

### 근거 (코드/메모리)
- ctx/gen 독립 TP·PP: `tensorrt_llm/llmapi/disagg_utils.py:171-211` (`extract_ctx_gen_cfgs`, instance_num_ranks=TP×PP×CP).
- 비대칭 막는 assert 없음 + KV 동질성만 검사: `CLAUDE.md` §프레임워크 판정 (`cacheFormatter.cpp::inquireSupport`).
- #14020 ctx-PP→gen-TP hang: release note KNOWN ISSUE (수정중 #15136). FlashInfer 금지(비대칭 TP 오출력 #6507) → `attn_backend: TRTLLM`.

---

## 4. 같은 노드 vs 다른 노드 — transport(UCX) + 멀티노드 분산

- **capability 게이트 아님.** same/cross 둘 다 UCX가 transport로 처리, 비대칭 여부와 무관.
  - same-node: UCX `cuda_copy/sm`(SHM). g5/g6/g6e는 **NVLink 없음** → cuda_ipc는 Nitro에서 막힐 수 있어 shm/cuda_copy로.
  - cross-node: UCX `tcp`. AWS xlarge/12xlarge는 **EFA(RDMA) 거의 없음** → TCP → 느림(TTFT 지배). → inter-node는 구조/correctness용, 깨끗한 성능은 intra-node(PCIe) 중심.
- **멀티노드 분산엔 Ray 아님 — TRT-LLM은 MPI.** 단, **독립 `trtllm-serve` 워커(우리 P1D1 방식)는 MPI도 불필요** (각 워커가 독립 프로세스, orchestrator가 HTTP/UCX로 조율). 한 인스턴스를 노드에 걸쳐 TP/PP로 펼치면 그때 MPI(`trtllm-llmapi-launch`/slurm)가 필요해짐. vLLM "PD엔 Ray 불필요"와 같은 결론, 메커니즘만 Ray→MPI.

---

## 5. 시나리오별 결론 (1P + "decode를 크게") — TRT-LLM은 PP도 후보

전제: prefill = 1 워커(TP1·PP1). "D를 어떻게 키우나"가 대칭/비대칭을 정함.

| 시나리오 | 분류 | 노드 | TRT-LLM 지금? | 방법/주의 |
|---|---|---|---|---|
| 다른노드 **DP** (D 복제 N) | DP, 대칭 | xPyD | ✅ | `num_instances:N` + orchestrator 라우팅 |
| 다른노드 **비대칭 TP** (D tp4) | 비대칭 TP | 2노드 | ✅ | ctx tp1 → gen tp4, UCX-TCP |
| 같은노드 **비대칭 PP** (D pp4) | 비대칭 PP | 1대(4GPU) | ⚠️ 실측 | assert 없음, Phase 0 확인 |
| **ctx-PP → gen-TP** | 비대칭 TP+PP | — | ⚠️ **hang** | #14020 — 300s 감시 |

**읽는 법**: vLLM에선 "PP 끼면 다 ❌"였지만, **TRT-LLM은 비대칭 TP·PP 둘 다 후보**다. 막히는 건 원리가 아니라 *특정 조합의 버그(#14020)* → Phase 0 게이트로 통과 조합만 본 스윕. 우리 목표("decode를 prefill보다 크게")는 **비대칭 TP(확실) 또는 PP(실측)** 로 달성.

> xPyD 라우팅: vLLM은 공식 proxy가 1P1D 한계였지만, **TRT-LLM `trtllm-serve disaggregated`는 `generation_servers.num_instances`>1 fan-out을 내장** → 별도 라우터 불필요.

---

## 6. per-side 측정 — prefill RPS/TPS vs decode RPS/TPS (vLLM MetricsScraper의 TRT-LLM 대응)

vLLM은 각 노드 `/metrics`의 누적 토큰 카운터를 1초 차분해 per-side를 구했다. **TRT-LLM은 메커니즘이 다르다 — 그대로 이식하면 깨진다.**

| 지표 | vLLM(차용 안 함) | **TRT-LLM 실제** |
|---|---|---|
| per-side **RPS** | `vllm:request_success_total` 차분 | **`trtllm_request_success_total`** 워커별 차분 ✅ (vLLM식 OK) |
| per-side **TPS** | `prompt/generation_tokens_total` 차분 | **누적 토큰 카운터 없음** → **RPS × 고정 ISL/OSL** (그리드가 길이 고정) |

**함정(핵심):**
1. 진짜 Prometheus는 워커 `/metrics`가 아니라 **`/prometheus/metrics`** (워커 `/metrics`는 Prometheus 텍스트가 아니라 **iteration stats JSON 리스트** → 스크레이퍼로 붙이면 깨짐). `return_perf_metrics:true`일 때만 마운트(우리 ctx/gen YAML 이미 켜둠).
2. orchestrator(:8000) `/metrics`는 **404** → 반드시 워커 포트(ctx :8001, gen :8011)로.
3. `numCtxTokens`/`numGenTokens`는 누적이 아니라 **per-iteration 순간값** → 차분 금지. (PyTorch 백엔드는 numGenTokens 자체 미기록.)
4. ctx 워커=prefill만, gen 워커=decode만 → **워커별로 따로 긁으면 side 분리는 자연히 됨.**

**측정 방식(채택):**
- per-side **RPS** = 워커별 `/prometheus/metrics`의 `trtllm_request_success_total` 1초 차분 + active-window 필터(vLLM 알고리즘 차용, 대상만 TRT-LLM).
- per-side **TPS** = `prefill_tps = prefill_rps × ISL`, `decode_tps = decode_rps × OSL` (그리드가 ISL/OSL 고정).
- **KV 전송시간**(side 분해의 보조) = orchestrator `/perf_metrics`의 `gen_perf_metrics[.perf_metrics].timing_metrics.kv_cache_transfer_end-start` (현 sweep.py가 수집).
- 근거: `tensorrt_llm/metrics/collector.py` (노출 메트릭 전부 = `trtllm_request_success_total` + e2e/ttft/tpot/queue 히스토그램, **토큰 카운터 없음**), `serve/openai_server.py:259-318`(/metrics=JSON, /prometheus/metrics 마운트, /perf_metrics), `:386-453`.

---

## 7. 부하/측정 도구 — 공식 최대 + per-side만 커스텀 (vLLM 공식 벤치 브랜치 철학)

- **공식 부하 도구**: `python -m tensorrt_llm.serve.scripts.benchmark_serving` — vLLM `benchmark_serving.py`의 fork, **공식 disagg slurm 벤치(`examples/disaggregated/slurm/benchmark/run_benchmark.sh`)가 호출**. orchestrator :8000 OpenAI를 침. token-id ISL 고정(`--random-ids --tokenize-on-client --random-range-ratio 0`)·`--ignore-eos`·OSL(`--random-output-len`)·Poisson(`--request-rate --burstiness 1.0`)·`--max-concurrency`·TTFT/TPOT/ITL/E2EL 전부 커버. (TPOT 정의도 우리 analyze.py와 동일.)
- `trtllm-bench`는 `--engine_dir` in-process라 serving 엔드포인트 못 침 → 부하 도구 아님.
- **권고**: 부하 코어=공식 benchmark_serving, **커스텀=per-side MetricsScraper + 디버깅(/perf_metrics·수집기)뿐**, sweep.py는 오케스트레이터(그리드·resume·S3·metadata) 유지.

---

## 8. 소스 인덱스 (TRT-LLM v1.2.1, 박아두기)
- cache_transceiver 설정: `tensorrt_llm/llmapi/llm_args.py:1807-1843` (CacheTransceiverConfig: backend Literal DEFAULT/UCX/NIXL/MOONCAKE/MPI)
- disagg ctx/gen 독립 TP·PP 파서: `tensorrt_llm/llmapi/disagg_utils.py:115-211`
- per-side 메트릭: `tensorrt_llm/metrics/collector.py`(Prometheus), `tensorrt_llm/serve/openai_server.py:259-318`(/metrics·/prometheus/metrics·/perf_metrics 마운트), `:386-453`
- 공식 부하: `tensorrt_llm/serve/scripts/benchmark_serving.py`; 호출 예 `examples/disaggregated/slurm/benchmark/run_benchmark.sh`
- 프레임워크 판정·#14020·dense DP축 없음: `disagg-exp/CLAUDE.md` §프레임워크 판정 (구 메모리 통합본)

---

## 9. 검증 방법 / TODO ("코드가 안 막음 ≠ 됨" → 확정)
- [ ] **Phase 0 게이트**: 대칭/비대칭 TP → PP 조합을 실제 기동/KV전송/출력정확성으로 실측 → `trtllm_support_matrix.md`. (특히 ctx-PP→gen-TP #14020 hang 300s 감시)
- [ ] **per-side RPS 스모크**: P1D1에서 ctx :8001·gen :8011 `/prometheus/metrics`에 `trtllm_request_success_total` 노출 확인 + 워커별 차분 = client RPS와 정합.
- [ ] **per-side TPS**: prefill_rps×ISL / decode_rps×OSL이 benchmark_serving의 시스템 throughput과 정합한지 한 점에서 대조.
- [ ] **KV전송시간 깊이**: `/perf_metrics`에서 `gen_perf_metrics.timing_metrics` vs `.perf_metrics.timing_metrics` 실제 키 깊이 확인(파서는 둘 다 대응).
- [ ] benchmark_serving 단발 호출로 ISL/OSL 고정·TPOT 정의 대조.
