# T1 — TRT-LLM PD 분리 결과

- 모델: Qwen/Qwen3-4B
- 토폴로지: 1P(tp1,pp1) + 1D(tp1,pp1)  ·  배치: inter-node  ·  KV전송: UCX
- 부하: 공식 benchmark_serving (open-loop, Poisson). point = p{prefill}_d{decode}_r{rate}

## 핵심 결과
| point | prefill_time_median_s | kv_transfer_time_median_s | decode_time_median_s | e2e_time_median_s | output_tokens_per_sec | decode_concurrent_reqs_mean | backlog_waiting_reqs_mean | prefill_minus_decode_completion_per_sec |
|---|---|---|---|---|---|---|---|---|
| p1024_d512_r1.0 | 0.156 | 0.541 | 34.101 | 35.373 | 470.0 | 25.6 | 29.5 | 0.021 |
| p1024_d512_r2.0 | 0.156 | 0.552 | 36.668 | 88.158 | 536.5 | 31.3 | 73.5 | 0.308 |
| p1024_d512_r4.0 | 0.157 | 0.553 | 36.623 | 124.607 | 542.7 | 32.0 | 80.7 | 0.307 |
| p2048_d128_r1.0 | 0.303 | 1.059 | 5.309 | 21.558 | 116.0 | 4.7 | 18.3 | 0.071 |
| p2048_d128_r2.0 | 0.304 | 1.047 | 5.360 | 89.208 | 119.1 | 4.9 | 30.6 | 0.110 |
| p2048_d128_r4.0 | 0.305 | 1.067 | 5.348 | 133.589 | 116.4 | 4.8 | 31.6 | 0.103 |

> 핵심표는 단계분해(초)·처리량·decode배치·backlog. 공식 TTFT/TPOT/E2EL·per-side rps·전체 수치 = `data.csv` · 시간순 = `plots/timeseries_*.png` · 포인트 비교 = `plots/compare_latency_decomp.png`(지연 단계분해)·`compare_perside.png`(per-side)

## 지표 뜻 / 출처·식
```
── 지표 = 무엇 / 출처 / 식  ([공식]=benchmark_serving · [서버]=perf_metrics·워커/metrics · [계산]=우리 코드). 시간=초 ──
[① 한 요청 시간 단계분해 — perf_metrics, 요청별 median]
prefill_time_median_s         [서버] ctx 첫토큰 − ctx 도착 (prefill 단계)
kv_transfer_time_median_s     [서버] kv_cache_transfer_end − start (inter-node KV전송 단계)
decode_time_median_s          [서버] gen 마지막토큰 − gen 첫토큰 (decode 단계, 첫토큰 이후)
e2e_time_median_s             [서버] gen 마지막토큰 − disagg 도착 (요청 전체) ≈ 위 셋의 합
   ※ warmup(8토큰, decode≤1s) 제외하고 측정 요청만 집계
[①b 단계 상세 — 한 요청 TTFT가 어디서 대기/소비되나 (perf_metrics 같은 시계)]
prefill_compute_p50_s   [서버] ctx first_token − first_scheduled (순수 prefill 연산, 작음·일정)
prefill_queue_p50_s     [서버] ctx first_scheduled − arrival (prefill 큐 대기; 고rate서 폭증=backpressure)
kv_transfer_p50_s       [서버] = 버퍼 대기 (inter-node KV전송, 작음)
handoff_p50_s           [서버] gen arrival − ctx first_token (orchestrator relay, 작음)
decode_queue_p50/mean/p99_s [서버] gen first_scheduled − arrival (decode 슬롯 대기; 보통 TTFT 지배)
ttft_recon_p50_s        [서버] disagg first_token − arrival (재구성 TTFT; 클라 ttft와 일치로 시계검증)
   ※ 단계 합 ≈ e2e (overlap 있어 정확 분할 아님). TPOT 재구성 = decode_p50_s / (출력토큰−1).
[② 공식 지연 — benchmark_serving, 클라이언트가 측정한 SLO]
requests_ok / fail_pct        [공식] 완료 요청수 / 실패율(%)
ttft_median_s / ttft_p99_s    [공식] 첫 토큰까지 (median / 99퍼센타일)
tpot_median_s                 [공식] 토큰당 시간(첫토큰 제외, median). ※E2E ≈ ttft + tpot×출력토큰수
e2e_client_p99_s              [공식] 요청 끝까지 (99퍼센타일)
output_tokens_per_sec         [공식] 생성토큰 합 / 측정시간
[③ prefill vs decode 요청 완료율·차이 — perf_metrics 절대 완료시각 기반(per-request)]
  ※prefill 완료=ctx 첫토큰(KV 준비됨), decode 완료=gen 마지막토큰. 각 측의 '첫완료~끝완료 구간'으로 나눔.
  ※1P1D라도 prefill이 더 좁은 구간에 다 끝내므로 prefill>decode로 갈라짐 → decode가 율속(병목)임이 드러남.
prefill_completion_rate_per_sec [계산] prefill 끝낸 요청수 ÷ (prefill 첫완료~끝완료 구간) = prefill이 초당 끝낸 요청수
decode_completion_rate_per_sec  [계산] decode 끝낸 요청수 ÷ (decode 첫완료~끝완료 구간) = decode가 초당 끝낸 요청수
prefill_minus_decode_completion_per_sec [계산] 위 둘의 차이 = prefill이 decode보다 빠른 정도(양수=decode 병목)
prefill_tokens_per_sec_approx  [계산·근사] prefill_completed × 입력길이(ISL). ※토큰 카운터 없어 곱셈 근사
decode_tokens_per_sec_approx   [계산·근사] decode_completed × 출력길이(OSL)
[④ 동시처리·KV·대기 — 워커 /metrics, 1Hz 샘플. mean=평균(ramp/idle 희석), max=peak(천장)]
prefill_concurrent_reqs_mean   [서버] prefill 워커 동시 처리 요청수(numContextRequests) 평균
decode_concurrent_reqs_mean/max[서버] decode 워커 동시 처리 요청수(numGenRequests) 평균/최대(=VRAM 천장)
decode_kvpool_used_pct_mean/max[서버] decode KV풀 사용률(usedNumBlocks/maxNumBlocks×100) 평균/최대
backlog_waiting_reqs_mean/max  [계산] (ctx_completed−gen_completed) 평균/최대 = decode 대기 중인 요청수(쌓인 양)
```

## 이 폴더 안내
```
REPORT.md      이 파일 — 핵심표 + 지표뜻 + 폴더안내
data.csv       전체 수치(모든 메트릭, RAW, 논문용)
plots/         timeseries_<pt>.png(포인트별 시간순) · compare_latency_decomp.png(단계분해 mean/p50/p99) · compare_perside.png(per-side rps·배치·backlog)
raw/           원본 JSON 전부(기록용): latency_throughput / perside_rps / kv_transfer /
               perside_batch_kv_backlog / timeseries.jsonl
metadata.json  설정(모델·ctx/gen TP·PP·placement)
.done_* .failed_*   재실행 스킵 마커
```
