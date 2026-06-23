# DEBUGGING — 디버그 토글·로그 구조·perf 메트릭 (TRT-LLM PD 분리)

> ⚠️ **측정(measured) 런에선 모든 디버그 로그를 OFF로.** 디버그 로그는 I/O 노이즈 → 변인 오염(research-rigor #7).
> 디버깅은 **smoke(실제 config + 적은 요청)** 에서만 켜고, 통과하면 끄고 본 스윕.

---

## 1. 빠른 디버깅 흐름 (smoke)
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

---

## 2. 로그 레벨 켜고 끄기

| 켜는 법 | 무엇 | 끄는 법 |
|---|---|---|
| `LOG_LEVEL=debug bash launch_trtllm.sh ...` | 워커 `--log_level` + orchestrator `-l` 한 번에 | `LOG_LEVEL` 생략(=info) |
| `export TLLM_LOG_LEVEL=debug` | TRT-LLM 파이썬 상세 로그(debug\|verbose\|trace) | `unset TLLM_LOG_LEVEL` |
| `export UCX_LOG_LEVEL=debug` | UCX(KV전송 전송계층) 상세 로그 | `unset UCX_LOG_LEVEL` |

`launch_trtllm.sh` 상단 **DEBUG 토글 블록**에 위 env가 주석으로 박혀 있음(주석 해제=ON, 재주석=OFF). 각 줄 끝에 "무슨 디버그인지" 명시.

---

## 3. 로그 구조 (어디에 뭐가 있나) — `$EXP_LOG_DIR/` (기본 `./results`)

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

---

## 4. KV전송 시간 / perf 메트릭 읽기

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

---

## 5. disagg 전용 디버그 env (각 무엇 / 끄는 법)

| env | 무엇을 하나 | 끄는 법 |
|---|---|---|
| `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` | KV전송 overlap 비활성 → 전송 타이밍 단순화(분리 측정) | `unset` |
| `UCX_TLS=tcp,cuda_copy,sm,self` | UCX 전송 경로 강제(EFA 없음 노드) | 기본값이라 보통 유지 |
| `UCX_LOG_LEVEL=debug` | UCX 핸드셰이크/전송 상세 | `unset` |
| `CACHE_BACKEND=UCX` (launch env) | KV 백엔드 — **단 실제값은 워커 extra YAML이 결정**(disagg YAML은 정보성) | extra YAML 수정 |

> 전송 백엔드를 정말 바꾸려면 `ctx/gen_extra_llm_api_options.yaml`의 `cache_transceiver_config.backend`를 직접 수정.

---

## 6. hang (무응답 300s+) 진단 — #14020
- 증상: 요청이 안 끝남, orchestrator 로그 멈춤.
- 1순위 의심: **ctx-PP → gen-TP 조합**(예 `CTX_PP=2 GEN_TP=2`). 알려진 hang(#14020).
- 조치: `trtllm_support_matrix.md`에 해당 셀 FAIL 기록 → 그 조합 제외, 또는 1.3.0rc 핀 검토.
- 확인: `LOG_LEVEL=debug` + `UCX_LOG_LEVEL=debug`로 KV전송 단계에서 멈추는지 추적.

---

## 7. 측정 전 체크 (디버그 OFF 확인)
- [ ] `LOG_LEVEL` 미설정(=info)
- [ ] `TLLM_LOG_LEVEL` / `UCX_LOG_LEVEL` unset
- [ ] `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP` unset (overlap = 정상 동작)
- [ ] `enable_autotuner` 끄지 않음(기본 on — 끄면 성능 저하)
- [ ] perf 수집은 ON 유지 OK (저오버헤드, 측정 후 1회 폴링)
