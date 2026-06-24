# KV-pool backpressure / crash — RCA 현황 + 재실험 런북

> 적대적 코드+데이터 RCA 결과(2026-06-25). **두 질문 모두 현재 데이터로는 NEEDS_REEXPERIMENT.**
> 핵심: "기존 메트릭이 인과를 못 박는다 + p128_d2048 데이터가 디스크에 아예 없다." 추측으로 단정 금지.

## 질문
- **Q1**: p1024_d512에서 prefill이 "요청큐에서 더 안 꺼내고 멈춘다(=context KV 풀이 꽉 차서 backpressure)"가 사실인가?
- **Q2**: p128_d2048은 왜 멈췄나(크래시)? 왜 기다리지 않고 assert 했나?

## 코드로 100% 확정된 사실 (CERTAIN)
1. 기본 정책 = **GUARANTEED_NO_EVICT** (`tensorrt_llm/llmapi/llm_args.py:1468`).
2. 이 정책은 블록 부족 시 **evict 안 하고 스케줄 루프를 BREAK(=대기)** (`cpp/.../capacityScheduler.cpp:322-325`). evict 경로 없음.
3. **disagg gen-init 요청도 context와 같은 게이트**를 통과 (`capacityScheduler.cpp:301-303`, `isDisaggGenerationInitState()` → `enoughAvailableBlocks()`).
4. 게이트는 **풀 시퀀스(promptLen + maxNewTokens) 블록을 선예약** (`kvCacheManager.cpp:2285-2287, 2315-2318`).
   → **즉 "decode가 maxNewTokens를 예약 안 해서 over-admit한다"는 내 가설은 코드상 거짓(REFUTED).**
5. 비게이트 per-step 할당 경로는 존재: `addToken → adjustBlocksIfNeeded → allocateBlock` → `TLLM_CHECK(hasFreeBlocks, "Can't allocate new blocks. No free blocks left.")` (`kvCacheManager.cpp:1508-1511, 1584`; receive-path add_sequence 1584, replaceSharedBlock 1732).

## 측정 데이터로 확정된 사실
- p1024_d512_r4: **context KV max 0.997 / mean 0.869**, decode KV max 0.985, decode_batch max 48, backlog max 109, **prefill_active_frac 1.0 (prefill_batch 1~5, 0인 샘플 없음)**.
- timeseries: backlog가 천장에서 **부드럽게 0으로 드레인**(꽉 찼다 풀리는 패턴 아님).
- **p128_d2048 데이터가 디스크에 없음.** crash 로그·"No free blocks left" 문자열은 소스 코드에만 존재, 어떤 로그/결과 파일에도 없음.

## 왜 NEEDS_REEXPERIMENT (무엇이 안 풀렸나)
- **Q1**: 대기 *메커니즘*과 *수치(99.7%)*는 진짜지만, "prefill이 멈춘다"는 *관측*은 데이터와 **모순**(prefill 한 번도 idle 아님). 게다가 `prefill_batch≈1` ↔ `context KV 99.7%`가 안 맞음(능동 prefill은 1개인데 풀은 꽉 참 = 풀을 채우는 건 "전송 대기 중인 held KV"일 가능성, 근데 그게 `numContextRequests`엔 안 잡힘). **per-sample context KV + 스케줄러 break 카운터가 없어 인과를 못 박음.** `backlog ≈ 63+48` 분해는 prompt-only(32블록)와 full(48블록) 회계를 섞은 거라 엄밀히 도출 아님.
- **Q2**: 내 크래시 가설(over-admit)은 코드로 반증됨. 풀-출력 예약을 하면 GUARANTEED_NO_EVICT는 **대기해야지 크래시하면 안 됨.** 그럼 *만약* 크래시한다면 다른 경로(예약 밖 할당): receive-path add_sequence, `num_extra_kv_tokens`(`resource_manager.py:491-493`), 또는 전송 중 context-pool 블록 점유. **크래시 자체가 미관측이라 미확정.**
- 단, **여전히 그럴듯한 가설(미증명)**: p128_d2048은 context:decode 용량비가 **~17배 불균형**(ctx 505 vs decode 29 요청). prompt 128tok=4블록이라 context는 안 막히고 → decode가 유일 병목 → decode 풀 오버플로. 핸드오프가 ctx free-block엔 게이트되지만 decode free-block엔 안 되는 게 의심 지점.

## ★ 한 번에 알아내는 재실험 (one-shot)
**같은 1ctx+1gen 토폴로지로 1세션, 2페이즈 연속**:
- Phase A = `p1024_d512 @ r4.0` (Q1 정상상태)  ·  Phase B = `p128_d2048 @ 같은 rate` (Q2 크래시 경계)
- **경계를 짧게 도달**시키려 풀을 작고 알려진 값으로: `free_gpu_memory_fraction` 낮추거나 `max_num_tokens`/`max_tokens`로 `maxNumBlocks`를 작게.

**계측 (대부분 프레임워크 기본 제공, 켜기만):**
1. **양쪽 워커** `TLLM_LOG_LEVEL=DEBUG`, stderr를 각각 `ctx.log`/`gen.log`로 분리.
2. **양쪽 워커** `/metrics kvCacheStats`를 ≥2Hz로 폴 → per-sample `{t, side, usedNumBlocks, maxNumBlocks, tokensPerBlock, freeNumBlocks}` JSONL. → 상수(2021/32/4/68) 실측 확정 + 지금 없는 per-sample **context KV 비율** 확보. *(harness: `metrics_sampler.py` trace에 `prefill_kv_frac`·`decode_kv_frac` 이미 추가됨 → 다음 런부터 기록.)*
3. `gen.log`에서 경고 verbatim 감시: `py_executor.py:1037-1041` **"num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"** + 타임스탬프.
4. 크래시 나면 `gen.log`에서 assert **"Can't allocate new blocks. No free blocks left."**의 **스택/발생 라인** 캡처 → 어느 호출부인지 구분(1584 receive add_sequence vs 1508-1511 per-step addToken vs 1732 replaceSharedBlock).
5. 기동 시 gen 워커의 resolved `capacity_scheduler_policy` 로그(DEBUG) + admission 시 gen-init당 예약 블록수 = `ceil((128+2048)/tokensPerBlock)`인지 확인(풀-출력 예약 증명).

**판정 규칙:**
- **Q1 CONFIRMED** ⟺ Phase A backlog 평평 구간에 (break 카운터 firing) **AND** ctx used/max ≈ 0.997 **AND** 같은 샘플에서 pending_queue>0 인데 admitted=0. → prefill이 ctx-KV로 실제 막힘. 반대로 admitted>0·break 없음이면 "ctx-full로 멈춘다"는 **falsified**(그냥 decode에 rate-matching).
- **Q2 RESOLVED** by Phase B: gen 정책이 GUARANTEED_NO_EVICT이고 예약이 풀-출력(128+2048)이면 over-admit설은 확정 거짓 → 크래시 발생 시 규칙4의 발생 라인이 **진짜 메커니즘**을 지목(예약 밖 할당). 크래시 안 나고 :1037 경고만 뜨며 backpressure하면 → 이 빌드선 p128_d2048이 크래시 안 함(원래 크래시는 다른 config였다는 뜻). admitted gen-init수 vs `floor(maxNumBlocks / ceil(2176/tokensPerBlock))`를 경고/assert 순간에 같이 찍어 과/저구독 정량화.

## 실행 커맨드 (복붙 — 원격 EC2에서)
하니스는 이미 resume(`.done`/`.failed` 마커)·양쪽 KV trace(`prefill_kv_frac`/`decode_kv_frac`)를 지원.
```bash
# 0) (선택) 크래시 경계 빨리 도달: ctx/gen yaml의 free_gpu_memory_fraction 낮춰 pool 작게.
#    measure 수도 줄여 빨리: export SWEEP_MEASURED_N=120
#    KV/큐 더 촘촘히 보려면: export SWEEP_SAMPLE_INTERVAL=0.5   (기본 1.0s)

# 1) 양쪽 워커 DEBUG 로그 ON (스케줄러/KV/경고/assert가 워커 로그에 찍히게)
export LOG_LEVEL=debug
export TLLM_LOG_LEVEL=debug
# (KV전송 의심되면) export UCX_LOG_LEVEL=debug

# 2) 서버 기동 (워커 로그 → $LOG_DIR/logs/trtllm_*_{ctx,gen}_*.log 로 자동 분리 저장)
bash launch_trtllm.sh ...        # 평소처럼 (ctx 노드 / gen 노드 각각)

# 3) sweep 재실행 — .done 있는 포인트는 자동 skip(=중단점부터 재개).
#    크래시했던 decode-heavy는 마커 지워 강제 재시도:
rm -f results/<CONFIG>/.failed_p128_d2048_*  results/<CONFIG>/.done_p128_d2048_*
python3 sweep.py ...             # p1024_d512(Q1) 와 p128_d2048(Q2) 둘 다 포함되게

# 4) 원인 자동 추출 (워커 로그 + trace에서 Q1/Q2 신호)
python3 diagnose_run.py --log-dir <그 런의 LOG_DIR>
python3 analyze.py --log-dir results --configs <CONFIG> --plot   # timeseries에 ctx KV·요청큐 패널 채워짐
```
`diagnose_run.py`가 출력: gen 정책(GUARANTEED_NO_EVICT?), "may not have enough kvCache" 경고 횟수,
크래시 assert 발생부(kvCacheManager.cpp:1584 vs :1508 vs :1732 → 진짜 메커니즘), 그리고
backlog 평탄 구간의 `ctx_kv` / `prefill_queue` 로 **Q1 자동 판정**(ctx_kv≈1 & 큐>0 ⇒ CONFIRMED).

## 수정 방향(코드 픽스, 참고)
decode admission을 풀-시퀀스 블록 예산으로 캡(예: d2048이면 ~29) **또는** gen→ctx backpressure를 *decode* free-block 기준으로(현재 ctx free-block 기준). — 단 이건 RCA 확정 후.
