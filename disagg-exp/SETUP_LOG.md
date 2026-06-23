# SETUP_LOG — 작업 로그 + provenance (나중에 정리/methods용)

> 시간순 행동 + 검증 증거 기록. 결정/규칙은 `CLAUDE.md`, 설계는 `EXPERIMENT_PLAN.md` 참조.
> 작업 위치: 로컬 Mac(파일·git). 런타임(서버/추론)은 아직 안 함 — 다음 단계(원격 GPU).

## 2026-06-10 — TRT-LLM fork 클론 + 브랜치 + 핀 검증 + 문서 스캐폴딩

### 1. Fork 생성 (GitHub UI)
- `NVIDIA/TensorRT-LLM` → fork **`ddps-lab/disaggregation-TensorRT-LLM`**.
- "Copy the main branch only" **체크 해제**(전체 브랜치+태그 가져옴). GitHub fork엔 "특정 버전만 fork" 기능 없음 → 전체 fork 후 태그에서 브랜치 따는 방식.

### 2. 로컬 클론
```bash
git clone --no-recurse-submodules https://github.com/ddps-lab/disaggregation-TensorRT-LLM.git
```
- 서브모듈 제외: 로컬 빌드 안 하고 **컨테이너로 실행**할 거라 소스 트리만 필요.
- 결과: 2.0G (서브모듈 포함 시 훨씬 큼).

### 3. git-lfs 문제 + 우회 (재현 시 참고)
- 증상: clone 중 `git-lfs: command not found` → `checkout failed` (exit 128). git-lfs 미설치 환경.
- LFS 콘텐츠(docs 이미지/바이너리)는 실험에 불필요 → 설치 대신 **필터 우회**:
```bash
git config --local filter.lfs.process ""
git config --local filter.lfs.smudge cat
git config --local filter.lfs.clean cat
git config --local filter.lfs.required false
git checkout -f HEAD          # LFS 파일은 포인터로 남고 워킹트리 완성 → status clean
```

### 4. 버전 핀 브랜치
```bash
git checkout -b disagg-exp/trtllm-v1.2.1 v1.2.1   # 태그 v1.2.1 기준
git push -u origin disagg-exp/trtllm-v1.2.1        # push 완료
```
- `git describe --tags` → `v1.2.1` (정확히 일치).
- `v1.2.1` 태그가 upstream 실재함 확인 (GitHub API: `repos/NVIDIA/TensorRT-LLM/tags` 에 `v1.2.1` 존재).

### 5. v1.2.1 소스 코드 검증 (릴리즈 노트 주장 ≠ 코드 증거 — research-rigor #3)
| 확인 항목 | 증거 (파일:라인) | 결과 |
|---|---|---|
| 버전 | `tensorrt_llm/version.py` → `__version__ = "1.2.1"` | ✅ |
| disagg 진입점 | `tensorrt_llm/commands/serve.py:646` `@click.command("disaggregated")`, `:679 def disaggregated`, `:962` 커맨드 매핑 | ✅ |
| Qwen3 dense | `tensorrt_llm/_torch/models/modeling_qwen3.py` (MoE 아닌 dense — Qwen3-4B용) | ✅ |
| KV 전송 | `cache_transceiver` 참조 `_torch/pyexecutor/py_executor.py` 등 | ✅ |
| disagg config 스키마 | `examples/disaggregated/disagg_config.yaml` | ✅ |
| Qwen3 disagg 예시 | `examples/configs/curated/qwen3-disagg-prefill.yaml` (단 MoE용: `enable_attention_dp`/`moe_expert_parallel_size`) | ✅ |
| 컨테이너 ref 패턴 | `nvcr.io/nvidia/tensorrt-llm/release:x.y.z` → 핀 `:1.2.1` | ✅ |

**핵심 발견**: `disagg_config.yaml`에서 `context_servers`/`generation_servers` 각각 독립 `tensor_parallel_size`·`pipeline_parallel_size` → 대칭/비대칭 자유. **`generation_servers.num_instances` = xPyD의 D 개수** (1P3D → `num_instances: 3`).

### 6. 문서 스캐폴딩 (코드 아님 — 사용자 방침 준수)
- `disagg-exp/CLAUDE.md` 신규 — 단일 진실원(목표·핀·dealbreaker·변인통제·스키마).
- `disagg-exp/EXPERIMENT_PLAN.md` — 기존 vLLM `../vllm-disaggregation/disagg-exp/`에서 복사.
- `disagg-exp/SETUP_LOG.md` — 이 파일.
- commit `96942817c1` (docs) → push 완료. (커밋은 `disagg-exp/trtllm-v1.2.1` 브랜치, default main 아님)

### 7. 글로벌 메모리 갱신
- `disagg-exp-experiment-overview.md`: "fork 이전 예정" → "셋업 완료"(repo 경로·브랜치·핀·검증사실).

## 2026-06-10 (2) — 하네스 코드 작성 + 소스 대조 검증 (방침 변경: Claude가 코드 작성)

### 8. 흐름: 소스 분석 → 작성 → 검증
- **(a) 리서치 워크플로우** (6 병렬 리더 → 빌드스펙): v1.2.1 소스에서 serve CLI·disagg 스키마·LLM args·Qwen3·예제·하네스 이식점을 코드레벨 확정. 핵심: 모델 positional(`--model` 없음), `--pp_size` 존재, `--dtype` CLI 없음, `enable_block_reuse`는 `kv_cache_config` 하위, orchestrator host/port는 disagg YAML에서만, sweep payload(token-id/ignore_eos/min_tokens) 전부 유효.
- **(b) 작성한 파일 8개** (모두 검증된 사실 기반):
  - 신규: `launch_trtllm.sh`(role별 trtllm-serve + disagg YAML 런타임 생성, env 파라미터화), `ctx_extra_llm_api_options.yaml`, `gen_extra_llm_api_options.yaml`, `disagg_config.yaml`(1P1D 정적), `trtllm_support_matrix.md`(Phase0 게이트), `README.md`(실행 가이드+vLLM대비 역할).
  - 이식: `sweep.py`(4곳), `analyze.py`(2곳), `setup.sh`(컨테이너 모델 재작성, 수집기 보존).
- **(c) 검증 워크플로우** (3 적대적 체커 vs 소스): **blocker 0, major 1, minor 11**(7개는 "정합 확인").
  - major: `launch_trtllm.sh` 빈배열 확장(`"${role_flag[@]}"`/`${PIDS[@]}`)이 bash<4.4(macOS)서 `unbound variable` → 가드(`[@]+...`/`[*]:-`/count check)로 수정. **로컬 bash 3.2.57서 안 깨짐 실증.**
  - minor 픽스: build_urls 공백트림, URL 개수 사전검증, CACHE_BACKEND/TP·PP가 URL모드선 inert(문서화), cuda_graph/server_role 주석 정확성.
- **로컬 정적검사 통과**: `bash -n`(launch/setup), `py_compile`(sweep/analyze), yaml.safe_load(3개).

## 현재 상태
- ✅ fork·브랜치(v1.2.1)·핀·문서 = 완료, origin 동기화.
- ✅ **하네스 코드 8개 작성 + 소스 대조 검증 완료** (blocker 0, 로컬 정적검사 통과).
- ⬜ 원격 GPU(g6e.12xlarge 등)에서 **Phase 0 스파이크 = 미시작** = 런타임 검증. (어떤 TP/PP 조합 OK인지, ctx-PP→gen-TP hang #14020 실측, max_batch_size/free_gpu_mem OOM 한계, server_role 필요 여부)

## 2026-06-10 (3) — 로깅·디버그·웜업 도구 추가 (vLLM 하네스 gap 이식)

### 9. 분석: vLLM 하네스 카탈로그 (5 Explore 에이전트)
- vLLM disagg-exp 9개 파일 정독 → gap 확정. 이미 이식된 것(수집기·S3·resume·2-phase·analyze 로직)과 빠진 것 분리.
- **핵심 발견**: TRT-LLM은 vLLM `instrumented_connector.py`(커스텀 KVConnector) 없이도 **네이티브 `/perf_metrics`로 per-request KV전송시간 제공**(더 풍부). 소스 확정: orchestrator `/perf_metrics`=`List[Dict]`(FIFO), `gen_perf_metrics.timing_metrics.kv_cache_transfer_start/end`(초, kv_cache_size>0일 때만). 활성=disagg YAML `perf_metrics_max_requests` + 워커 `return_perf_metrics:true`.
- **웜업 조사**: 서버측 웜업은 TRT-LLM 자동(graph/autotuner, `py_executor.py:276-287`). 클라 2-phase 웜업은 여전히 필수(disagg UCX 첫 연결 lazy, `kv_cache_transceiver.py`). → WARMUP 10→20.

### 10. 구현 (사용자 결정 반영)
- **KV계측**: sweep `fetch_perf_metrics()`→`perf_*.json`, analyze `load_perf()`+`kv_p50/p99` 컬럼. ctx/gen YAML `return_perf_metrics:true`, disagg YAML `perf_metrics_max_requests:1000`.
- **디버그 토글**: launch `LOG_LEVEL`(워커 --log_level/orch -l) + DEBUG 토글 블록(각 env 인라인 주석 "무엇/끄는법"), `DEBUGGING.md` 신규. 측정 OFF 강조(변인통제).
- **smoke**: sweep `SWEEP_PD_PAIRS` env(단일포인트). 실제 config + 적은 요청 절차 문서화.
- **웜업**: `WARMUP_N` 10→20 + 서버/클라 웜업 구분 문서화.
- **CUDA graph OFF**: gen YAML `cuda_graph_config: null`(사용자 결정, 균일 eager).
- 문서: CLAUDE(변인통제·파일맵)·README(smoke·산출물·5절)·EXPERIMENT_PLAN·LEARNING_NOTES(F.웜업)·DEBUGGING.md.
- 정적검사 통과(bash -n/py_compile/yaml/PD파서). **적대적 검증 후 커밋.**

## 2026-06-10 (4) — 문서 정리 (중복제거 + 이름수정, 내용 보존)
- **이름수정**: `병렬화-커넥터.md` → `병렬화-KV전송-측정.md`. 이유: TRT-LLM엔 vLLM 같은 "커넥터" 개념이 없고 cache_transceiver를 쓰므로 파일명이 내용과 불일치(vLLM 시절 잔재). 본문 H1은 원래 맞았음.
- **중복제거(정본 일원화)**: 변인통제·dealbreaker·disagg 스키마·프레임워크판정의 **정본 = CLAUDE.md 한 곳**. EXPERIMENT_PLAN의 ②클라/③메트릭/④인프라 재서술과 §리스크는 CLAUDE 포인터로 축약(고유분=Phase 0~3·그리드·vLLM→TRT 매핑표·하드웨어현실 유지). LEARNING_NOTES/DEBUGGING의 dealbreaker는 각자 역할 각도(교육/디버그)만 남김.
- **파일맵 정정**: CLAUDE.md 파일맵을 6종(stale)→8종 표로 교체 = 정본 문서 인덱스(역할·언제 여나·정본항목). README 파일-역할 표는 코드파일용이라 유지.
- 산출물·코드 변경 없음(문서만). 문서 8종 유지, 역할 명확화 + 중복 0 목표.

## 2026-06-10 (5) — per-side 측정 구현 (소스 전수조사 → 검증 → 코드)
- **조사**(9-에이전트 워크플로우, 411k 토큰): v1.2.1 소스에서 원하는 8개 메트릭(prefill/decode RPS·TPS + 전체 TTFT·TPOT·throughput·latency)의 공식-vs-커스텀을 코드단 확정. 핵심 3주장 적대적 검증.
  - **결과**: 전체 4개 = 공식(우리 sweep이 이미 동일 정의로 측정). per-side RPS = orchestrator :8000 `/prometheus/metrics`의 `ctx_/gen_completed_requests_total`(단일 엔드포인트, role 접두) 윈도우 차분 = 거의 공식. per-side TPS = **유일 공식불가**(토큰 카운터 0개 confirmed) → RPS×고정길이 파생.
  - **검증 정정**: ① benchmark_serving E2EL은 기본 출력에서 빠짐(`--percentile-metrics`에 e2el 필수) — 단 우리 sweep은 e2e_s 직접 측정해 무관. ② per-side RPS 워커 카운터 차분 confirmed + orchestrator 단일 엔드포인트가 더 깔끔.
- **구현**(3파일): `prom_scrape.py` 신규(Prometheus 텍스트 파서 + orchestrator 스냅샷, graceful), `sweep.py`(run_point에 `prom_out` — measured 윈도우 전/후 스냅샷 → `prom_*.json`, warmup 제외), `analyze.py`(`load_prom` — RPS=Δ/window, TPS=RPS×mean(token), 표 컬럼 `pf_rps/dc_rps/pf_tps/dc_tps`).
- **검증**: py_compile(3.9) + 기능테스트(3.11, aiohttp/numpy 스텁): 파서·_pick(_total 자동매칭)·load_prom(RPS/TPS 산식)·graceful(파일없음/None/window0)·토큰길이 폴백 전부 PASS. **런타임(카운터 실노출)은 GPU 스모크에서**(병렬화-KV전송-측정.md §9).
- 문서: 병렬화-KV전송-측정.md §6/§7/§9 + CLAUDE 파일맵 + README(§1 prom_scrape·§5 prom_*.json·분석컬럼).

## 2026-06-11 — 부하코어 교체: 손수 짠 aiohttp → 공식 benchmark_serving
- **동기**: sweep.py가 공식 서버를 호출하긴 하나 부하·측정 *코드는 손수 짠 aiohttp*(공식 benchmark_serving 미호출, grep으로 import 0 확인). 사용자 결정 = "공식 위주" 철학상 진짜 공식 도구로 교체.
- **조사**(Explore): v1.2.1 benchmark_serving 정확한 CLI/플래그·result.json 키·warmup 부재를 소스 확정(argparse :1019-1407, result.json :495-599, backend_request_func :309-314 `--random-ids`+`--tokenize-on-client`→`prompt_token_ids`).
- **구현**:
  - `sweep.py` 전면 개편: `Result`/`_do_request`/aiohttp `fire_phase` 제거 → `BENCH_MODULE`+`build_bench_args`+`run_bench`(asyncio.create_subprocess_exec). run_point = warmup(소량 `--non-streaming`) → before 스냅샷 → measured(`--save-result --save-detailed` → `bench_<point>.json`) → after 스냅샷 → `prom_<point>.json`. 오케스트레이션(그리드·resume·S3·metadata·perf·health) 유지. metadata에 `load_tool: benchmark_serving`.
  - `analyze.py`: `load_points`+`analyze_point`(우리 계산) 제거 → `load_bench`(공식 result.json의 p50/p99_{ttft,tpot,itl,e2el}_ms·output_throughput 읽기). `load_prom` 시그니처 rows→mean_pt/mean_ct. 표 컬럼 = ttft/tpot/**itl/e2el**/out_tok·s + kv + per-side. `$/Mtok`=output_throughput.
  - `prom_scrape.py` 변경 없음(서브프로세스를 bracket).
- **변인통제**: `--percentile-metrics ttft,tpot,itl,e2el`(e2el 필수)·`--random-range-ratio 0`(길이 고정)·warmup은 별도 호출.
- **검증**: py_compile(3) + 기능테스트(3.11): `_split_host_port`·`build_bench_args`(measured/warmup 플래그)·`load_bench`(키 매핑·fail_rate·mean_len)·`load_prom`·`print_table`(NO DATA 혼합)·`$/Mtok` 전부 PASS. **런타임은 GPU smoke**(benchmark_serving 실호출·temperature 기본값·tokenizer 로드).
- 문서: 병렬화-KV전송-측정 §7·CLAUDE(파일맵·변인통제)·README(§1·§5·smoke·§B-0)·DEBUGGING·EXPERIMENT_PLAN 갱신.

## 아직 안 한 것 / 주의 (코드에서 확정 못 함 → GPU에서)
- 로컬에서 TRT-LLM **빌드/실행 안 함** (컨테이너로 원격에서). 코드 정확성은 정적, **동작 검증은 GPU**.
- 컨테이너 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` 실제 pull·기동, Qwen3-4B 로드, KV전송 동작, 출력정확성(비분리 비교) = 전부 Phase 0.
- COST_PER_HR(analyze.py) 단가는 최종 인스턴스/리전 확정 후 채울 것 (현재 placeholder).
