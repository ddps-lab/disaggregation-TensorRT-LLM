# KV-pool backpressure / crash — RCA 현황 + 재실험 시 원인 읽는 법

> 적대적 코드+데이터 RCA 결과(2026-06-25). **두 질문 모두 현재 데이터로는 NEEDS_REEXPERIMENT.**
> 핵심: "기존 메트릭이 인과를 못 박는다 + p128_d2048 데이터가 디스크에 아예 없다." 추측으로 단정 금지.

## 질문
- **Q1**: p1024_d512에서 prefill이 "요청큐에서 더 안 꺼내고 멈춘다(=context KV 풀이 꽉 차서 backpressure)"가 사실인가?
- **Q2**: p128_d2048은 왜 멈췄나(크래시)? 왜 기다리지 않고 assert 했나?

## 코드로 100% 확정된 사실 (CERTAIN)
1. 기본 정책 = **GUARANTEED_NO_EVICT** (`tensorrt_llm/llmapi/llm_args.py:1468`).
2. 이 정책은 블록 부족 시 **evict 안 하고 스케줄 루프를 BREAK(=큐에서 더 안 꺼냄)** (`cpp/.../capacityScheduler.cpp:322-325`). evict 경로 없음.
3. **disagg gen-init 요청도 context와 같은 게이트**를 통과 (`capacityScheduler.cpp:301-303`, `isDisaggGenerationInitState()` → `enoughAvailableBlocks()`).
4. 게이트는 **풀 시퀀스(promptLen + maxNewTokens) 블록을 선예약** (`kvCacheManager.cpp:2285-2287, 2315-2318`).
   → **즉 "decode가 maxNewTokens를 예약 안 해서 over-admit한다"는 가설은 코드상 거짓(REFUTED).**
5. context는 prefill KV를 `ctx_in_transmission_requests`에 잡고 **transfer 완료 때만 해제** (`py_executor.py:2707-2728`); transfer는 gen이 "fitting" gen-init을 스케줄할 때만 트리거(`_recv_disagg_gen_cache`, `py_executor.py:2172`).
   → Q1 체인(decode 꽉참 → gen pull 안 함 → context held KV 안 풀림 → context 풀 꽉참 → context BREAK)은 **코드상 유일한 downstream 결합 경로**.
6. 비게이트 per-step 할당 경로 존재: `addToken → adjustBlocksIfNeeded → allocateBlock` → `TLLM_CHECK(hasFreeBlocks, "Can't allocate new blocks. No free blocks left.")` (`kvCacheManager.cpp:1508-1511, 1584`; receive add_sequence 1584, replaceSharedBlock 1732).

## 측정 데이터로 확정된 사실
- p1024_d512_r4: **context KV max 0.997 / mean 0.869**, decode KV max 0.985, decode_batch max 48, backlog max 109, prefill_batch 1~5(0인 샘플 없음).
- timeseries: backlog가 천장에서 **부드럽게 0으로 드레인**.
- **p128_d2048 데이터가 디스크에 없음.** crash 로그·"No free blocks left" 문자열은 소스에만 존재.

## 왜 아직 NEEDS_REEXPERIMENT
- **Q1**: 메커니즘·수치는 진짜지만, "prefill이 멈춘다"는 *관측*이 데이터와 긴장(prefill_batch 한 번도 0 아님인데 ctx KV 99.7%). throttle이 prefill 멈춤이 아니라 **admit 전 "큐 대기"로** 나타나기 때문일 가능성. per-sample context KV·admit/break 이벤트가 없어 런타임 인과를 못 박음.
- **Q2**: 크래시 자체가 미관측(데이터 없음). over-admit 가설은 코드로 반증. *만약* 크래시한다면 예약 밖 할당 경로(receive add_sequence / `num_extra_kv_tokens` / 전송 중 ctx 블록 점유)일 텐데, 로그 없이 확정 불가.
- 그럴듯한 미증명 그림: p128_d2048은 ctx:decode 용량비 ~17배 불균형(ctx 505 vs decode 29). prompt 4블록이라 ctx 안 막히고 decode가 유일 병목 → decode 풀 오버플로.

---

## ★ 재실험 때 "터진 원인"을 100% 확신으로 읽는 법 — 코드/스크립트 불필요
에러난 decode-heavy 포인트(p128_d2048 류)를 다시 돌리면 gen 워커가 또 abort한다.
그때 **assert가 스스로 워커 로그(stderr)에 메시지 + C++ 스택을 찍는다 — 이게 곧 100% 원인이다.**
따로 뽑는 스크립트 필요 없음. 그냥 그 로그를 읽으면 된다.

**준비(설정 한 줄, 코드 아님):** 기동 전 로그레벨만 올린다 → 스택/주변 맥락이 자세해짐.
워커 로그는 `launch_trtllm.sh`가 이미 `$LOG_DIR/logs/trtllm_*_{ctx,gen}_*.log`로 자동 저장.

**터지면 gen 로그에서 이 줄을 찾는다:** `Can't allocate new blocks. No free blocks left.`
**바로 뒤 스택의 발생 라인이 = 진짜 메커니즘:**
- `kvCacheManager.cpp:1584` (add_sequence, **전송 받을 때** 할당) → gen 풀에 KV 받을 자리 없음 = receive-path. = "decode 못 받는데 핸드오프가 밀어붙여 받다 터짐."
- `kvCacheManager.cpp:1508~1511` (addToken, **decode 토큰 자랄 때** 할당) → 예약분 넘겨 자라다 터짐 = decode 성장.
- `kvCacheManager.cpp:1732` (replaceSharedBlock) → block_reuse 경로.
같은 스택/로그에 보통 `usedNumBlocks / maxNumBlocks` 도 찍혀 풀 크기·점유가 그대로 나온다.

**보조 신호(이미 로그에 있음, 추가코드 X):**
- gen 로그의 `num_fitting_reqs=0 ... may not have enough kvCache` 경고(`py_executor.py:1037`) = decode가 들어오는 요청을 못 받고 있다는 신호. 이게 assert 직전에 뜨면 "decode 풀 가득 → 못 받음"이 그림.
- 기동 로그의 `capacity_scheduler_policy` = GUARANTEED_NO_EVICT 확인(풀-출력 예약 ⇒ over-admit 아님 재확인).

**판정:**
- 위 스택 발생 라인 = 예약 밖 할당이 어디서 나는지 = 진짜 원인. (1584면 receive, 1508~면 decode성장, 1732면 reuse.)
- 만약 **크래시 안 나고** 경고만 뜨며 backpressure하면 → 이 빌드/설정선 p128_d2048이 안 터짐(원래 크래시는 다른 config였다는 뜻).

**재실행:** 하니스 기본 — `.done` 포인트는 자동 skip(=빠진 것부터 재개), 크래시했던 포인트는 `results/<CONFIG>/.failed_<point>*` 마커만 지우면 재시도.

## 수정 방향(원인 확정 후 참고)
decode admission을 풀-시퀀스 블록 예산으로 캡 **또는** gen→ctx backpressure를 *decode* free-block 기준으로(현재 ctx free-block 기준).
