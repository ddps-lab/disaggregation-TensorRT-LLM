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

## 6. per-side 측정 — prefill RPS/TPS vs decode RPS/TPS (소스 전수조사 + 적대적 검증 완료 2026-06-10)

vLLM은 각 노드 `/metrics`의 누적 토큰 카운터를 1초 차분해 per-side를 구했다. **TRT-LLM은 메커니즘이 다르다 — 그대로 이식하면 깨진다.** 9-에이전트 워크플로우로 v1.2.1 소스를 전수조사하고 핵심 3주장을 적대적으로 검증한 결과:

| 지표 | 공식 가능? | **TRT-LLM 실제 소스** |
|---|---|---|
| per-side **RPS** | 🟡 부분공식 | **orchestrator :8000 `/prometheus/metrics`의 `ctx_completed_requests_total` / `gen_completed_requests_total`** (단일 엔드포인트, role 접두사) 윈도우 차분. 대안=워커별 `trtllm_request_success_total` |
| per-side **TPS** | ❌ 공식불가(검증 confirmed) | **누적 토큰 카운터가 어디에도 없음** → **RPS × 고정 ISL/OSL**(그리드가 길이 고정, 우리가 강제) |

**핵심 발견 (vLLM 대비 더 깔끔):**
- **orchestrator(:8000)가 per-side 카운터를 단일 엔드포인트로 노출**한다. `instance_metric()`이 role별 접두사(`ctx`/`gen`)를 붙여 `completed_requests` Counter를 만들고(perf_metrics.py:93-111), prometheus_client이 `_total`을 자동 부착 → `ctx_completed_requests_total`/`gen_completed_requests_total`. ctx=prefill 완료, gen=decode 완료를 각각 셈(openai_client.py:267 `.inc()`). **워커별로 돌아다닐 필요 없음.**
- per-side **TPS는 진짜 공식 불가**: `/prometheus/metrics`엔 `request_success_total` + 4개 latency 히스토그램뿐, **토큰 카운터 0개**(collector.py:27-68). 워커 `/perf_metrics` per-request 레코드에도 토큰 필드 없음(timing/kv만, openai_server.py:414-452). → 토큰수는 **우리가 강제한 길이**(token-id prompt=prefill_len, max_tokens+ignore_eos=decode_len)로 곱해 파생, `usage.prompt/completion_tokens`로 검증.

**함정(정정 포함):**
1. orchestrator(:8000)엔 `/metrics`가 **없음(404)** 이지만 **`/prometheus/metrics`는 있음**(openai_disagg_server.py:156-158) — 여기에 ctx_/gen_ 카운터 족(族)이 노출됨. *(과거 메모: "orchestrator는 워커 포트로만" → 정정: per-side RPS는 orchestrator 단일 엔드포인트가 1순위.)*
2. 워커 `/metrics`(JSON iteration stats)와 워커 `/prometheus/metrics`(진짜 Prometheus, `return_perf_metrics:true`일 때만)는 다름 — 후자만 스크레이프. `numCtxTokens`/`numGenTokens`는 per-iteration 순간값이라 차분 금지.
3. 카운터는 워커가 `return_perf_metrics`로 떠야 생김(openai_server.py:138). ctx/gen extra YAML에 이미 켜둠.

**측정 방식(채택):**
- per-side **RPS** = orchestrator `/prometheus/metrics`를 **measured 윈도우 시작/끝에 1회씩 스냅샷** → `(end−start)/window_s`. (warmup 제외 = 변인통제). 1순위 키 `ctx_/gen_completed_requests_total`, 폴백 = 워커 `trtllm_request_success_total` 합.
- per-side **TPS** = `prefill_tps = prefill_rps × mean(prompt_tokens)`, `decode_tps = decode_rps × mean(completion_tokens)` (analyze.py, ~5줄).
- **전체 TTFT/TPOT/throughput** = 우리 sweep.py가 이미 측정(공식 benchmark_serving과 정의 동일). **전체 latency(E2EL)**도 sweep `e2e_s`로 직접 측정 — *공식 benchmark_serving은 E2EL이 기본 출력에서 빠짐*(§7).
- **KV 전송시간**(side 분해 보조) = orchestrator `/perf_metrics`의 `gen_perf_metrics[.perf_metrics].timing_metrics.kv_cache_transfer_end−start` (현 sweep.py 수집).
- 근거: perf_metrics.py:71,92-113(role 접두 카운터), openai_client.py:267(.inc), collector.py:27-68(워커 메트릭=토큰카운터 0), openai_server.py:138,414-452, openai_disagg_server.py:156-158.

---

## 7. 부하/측정 도구 — 공식 최대 + per-side만 커스텀 (vLLM 공식 벤치 브랜치 철학)

- **공식 부하 도구**: `python -m tensorrt_llm.serve.scripts.benchmark_serving` — vLLM `benchmark_serving.py`의 fork, **공식 disagg slurm 벤치(`examples/disaggregated/slurm/benchmark/run_benchmark.sh`)가 호출**. orchestrator :8000 OpenAI를 침. token-id ISL 고정(`--random-ids --tokenize-on-client --random-range-ratio 0`)·`--ignore-eos`·OSL(`--random-output-len`)·Poisson(`--request-rate --burstiness 1.0`)·`--max-concurrency` 지원.
- ⚠️ **검증으로 잡은 함정 — E2EL은 기본 출력에서 빠짐**: 결과 JSON에 throughput(무조건)·TTFT·TPOT는 기본(`--percentile-metrics`=`ttft,tpot,itl`)으로 나오지만 **E2EL(전체 latency)는 안 나옴**. `--percentile-metrics ttft,tpot,itl,e2el`을 줘야 `*_e2el_ms`가 출력됨(benchmark_serving.py:537-539, 공식 run_benchmark.sh:71은 이미 e2el 포함). → 공식 도구를 쓸 땐 이 플래그 필수. **단 우리 sweep.py는 `e2e_s`를 직접 재므로 무관.**
- **단일 엔드포인트 한계**: benchmark_serving은 한 base-url만 침(benchmark_serving.py:703-708) → **per-side를 못 냄**(전체 RPS만). per-side는 별도 스크레이퍼 필요.
- `trtllm-bench`는 `--engine_dir` in-process라 serving 엔드포인트 못 침 → 부하 도구 아님.
- **채택 구조**: sweep.py를 **부하코어 겸 오케스트레이터로 유지**(전체메트릭을 공식과 정의-동일하게 이미 측정 + measured 윈도우 전후로 per-side 스크레이프를 끼워넣을 수 있음 — 단일엔드포인트 benchmark_serving은 못 하는 것). 커스텀=**per-side 스크레이퍼(`prom_scrape.py`) + analyze per-side**뿐. 공식 benchmark_serving은 **선택적 교차검증**(`sweep_official.py`)으로 전체메트릭 일치 확인(논문 비교가능성).

---

## 8. 소스 인덱스 (TRT-LLM v1.2.1, 박아두기)
- cache_transceiver 설정: `tensorrt_llm/llmapi/llm_args.py:1807-1843` (CacheTransceiverConfig: backend Literal DEFAULT/UCX/NIXL/MOONCAKE/MPI)
- disagg ctx/gen 독립 TP·PP 파서: `tensorrt_llm/llmapi/disagg_utils.py:115-211`
- per-side 메트릭: `tensorrt_llm/metrics/collector.py`(Prometheus), `tensorrt_llm/serve/openai_server.py:259-318`(/metrics·/prometheus/metrics·/perf_metrics 마운트), `:386-453`
- 공식 부하: `tensorrt_llm/serve/scripts/benchmark_serving.py`; 호출 예 `examples/disaggregated/slurm/benchmark/run_benchmark.sh`
- 프레임워크 판정·#14020·dense DP축 없음: `disagg-exp/CLAUDE.md` §프레임워크 판정 (구 메모리 통합본)

---

## 9. 검증 방법 / TODO ("소스가 정의함 ≠ 런타임에 노출됨" → GPU 스모크로 확정)
소스는 확정(고신뢰). 남은 건 **런타임 노출**뿐 — P1D1 스모크에서 `curl :8000/prometheus/metrics`로 실측:
- [ ] **orchestrator per-side 카운터 실재 확인**: `ctx_completed_requests_total`·`gen_completed_requests_total`이 **non-zero로 노출**되나(role 접두 `ctx`/`gen` 적용, 단일 uvicorn에서 멀티프로세스 분실 없는지). 빠지면 폴백=워커 `trtllm_request_success_total`.
- [ ] **global vs per-worker**: ctx_/gen_completed가 role 전체 집계인지 워커별인지 → **1P3D**에서 decode RPS를 인스턴스별로 봐야 하면 워커별 `trtllm_request_success_total`로.
- [ ] **ctx↔gen 1:1**: 분리에선 요청 1개=ctx 1회+gen 1회 → 정상상태에서 `ctx_completed≈gen_completed`인지(재시도/다중응답 인플레 없는지) → `TPS=RPS×len` 성립 확인.
- [ ] **per-side TPS 정합**: `decode_rps×decode_len`이 전체 output throughput과 noise 내 일치하는지 한 점 대조.
- [ ] **윈도우 정렬**: 카운터 스냅샷 wall-clock 간격 vs sweep send-타임스탬프 윈도우가 achieved_rate와 noise 내 정합.
- [ ] **(선택) ctx_/gen_ latency 히스토그램**(`gen_first_token_latency_seconds` 등 openai_client.py:186,237,241,251 `.observe()`)이 non-zero `_count/_sum`이면 per-side TTFT/TPOT 공식 교차검증 가능.
- [ ] **Phase 0 게이트**(병행): TP·PP 조합 실측 → `trtllm_support_matrix.md` (ctx-PP→gen-TP #14020 hang 300s 감시).
