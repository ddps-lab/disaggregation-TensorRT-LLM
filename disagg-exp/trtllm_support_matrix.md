# TRT-LLM v1.2.1 PD-disagg 지원 매트릭스 (Phase 0 게이트 산출물)

> **목적**: 대량 스윕 전, "어떤 (TP,PP) 조합이 실제로 안 깨지고 정확히 도는가"를 실측으로 확정.
> 각 셀이 **3개 다 PASS**여야 Phase 1 config 매트릭스에 포함. (research-rigor #8 spike-before-scale)
> 환경: multi-GPU 1대 (g6e.12xlarge=4×L40S 권장). 컨테이너 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`.

## 판정 기준 (셀마다 3개)
1. **기동(launch)**: ctx+gen+orchestrator 전부 `/health` OK, 300s 내 (hang 아님).
2. **KV전송(transfer)**: 첫 요청이 prefill→decode로 완주, 토큰 정상 생성.
3. **출력정확성(correctness)**: 비분리(단일 trtllm-serve) 동일 prompt+seed(temp=0) 출력과 일치 (garbled/오출력 없음).

## 측정 방법 (요약)
```bash
# 1) 비분리 baseline (정확성 비교 기준)
CUDA_VISIBLE_DEVICES=0,1 trtllm-serve Qwen/Qwen3-4B --backend pytorch --tp_size 2 --port 9000 \
    --extra_llm_api_options gen_extra_llm_api_options.yaml
#    → /v1/completions에 temp=0로 고정 prompt 던져 출력 저장

# 2) disagg (조합별) — launch_trtllm.sh로 기동
LABEL=spike_ctxTP2_genTP2 CTX_TP=2 GEN_TP=2 NUM_CTX=1 NUM_GEN=1 \
    bash launch_trtllm.sh all
#    → 같은 prompt+temp=0 출력이 baseline과 일치하는지 비교

# hang 감시: orchestrator 로그/요청에 300s 타임아웃. 무응답이면 FAIL(hang)로 기록.
```

## 매트릭스 (실측 후 채울 것)

| # | context (TP,PP) | generation (TP,PP) | placement | 기동 | KV전송 | 정확성 | 비고 |
|---|---|---|---|---|---|---|---|
| 대칭 TP | (2,1) | (2,1) | intra | ⬜ | ⬜ | ⬜ | 견고 예상 |
| 비대칭 TP↓ | (2,1) | (1,1) | intra | ⬜ | ⬜ | ⬜ | head 재매핑 |
| 비대칭 TP↑ | (1,1) | (2,1) | intra | ⬜ | ⬜ | ⬜ | head 재매핑 |
| 대칭 PP | (1,2) | (1,2) | intra | ⬜ | ⬜ | ⬜ | PP-in-PD |
| 비대칭 PP↓ | (1,2) | (1,1) | intra | ⬜ | ⬜ | ⬜ | gather 방향 |
| 비대칭 PP↑ | (1,1) | (1,2) | intra | ⬜ | ⬜ | ⬜ | scatter 방향 |
| ⚠️ ctx-PP→gen-TP | (1,2) | (2,1) | intra | ⬜ | ⬜ | ⬜ | **#14020 hang 위험 — 300s 감시** |
| 1P1D 베이스 | (1,1) | (1,1) | intra | ⬜ | ⬜ | ⬜ | sanity |
| 1P1D inter | (1,1) | (1,1) | inter | ⬜ | ⬜ | ⬜ | TCP, 느림 예상 |

(범례: ✅ PASS / ❌ FAIL / ⬜ 미측정. FAIL이면 비고에 에러·로그 경로.)

## 게이트 결론 (Phase 0 완료 후 작성)
- 통과 조합: _____
- 막힌 조합 + 원인: _____
- PP가 광범위하게 막히면 → 사용자와 상의 (1.3.0rc 핀 변경 vs 해당 축 제외).
- ⚠️ 필수 확인: 모든 셀에서 `attn_backend: TRTLLM` 적용됐는지(FlashInfer 폴백 아님), 비대칭 TP 오출력(#6507) 회귀 없는지.
