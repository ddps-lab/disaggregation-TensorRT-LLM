"""per-side Prometheus 스크레이퍼 — prefill/decode를 따로 재는 유일한 커스텀 조각.

배경 (소스 전수조사 + 적대적 검증, 2026-06-10 — 병렬화-KV전송-측정.md §6):
  - 전체 메트릭(TTFT/TPOT/throughput/latency)은 sweep.py가 이미 측정(공식 benchmark_serving과 정의 동일).
  - per-side RPS만 공식 도구가 못 줌 → 이 스크레이퍼가 필요.
  - per-side TPS는 공식 토큰 카운터가 *어디에도 없어*(collector.py:27-68에 토큰 카운터 0개,
    워커 /perf_metrics에도 토큰 필드 없음) RPS×고정길이로 analyze.py가 파생.

공식 소스 (단일 엔드포인트 = orchestrator :8000 /prometheus/metrics):
  - orchestrator가 role 접두(ctx/gen) Counter를 노출. instance_metric()이
    `{ctx|gen}_completed_requests`를 만들고(perf_metrics.py:93-111), 요청 완료 때 .inc()
    (openai_client.py:267). prometheus_client이 Counter에 `_total`을 자동 부착
    → 노출 시리즈 = `ctx_completed_requests_total` / `gen_completed_requests_total`.
  - ctx=prefill 완료수, gen=decode 완료수. measured 윈도우 전/후로 1회씩 스냅샷 →
    (after-before)/window_s = per-side RPS. (워커별로 돌아다닐 필요 없음.)

폴백(노출 안 될 때): 워커별 `trtllm_request_success_total`(collector.py:27-30) 합.
  단 1P1D에선 orchestrator 집계 = 워커값이라 orchestrator만으로 충분. 1P3D에서 decode를
  인스턴스별로 봐야 하면 그때 워커 스크레이프 추가(GPU 스모크에서 결정 — §9 TODO).

robust 설계: HTTP 실패/미노출이면 None을 담아 측정을 막지 않음(graceful, fetch_perf_metrics와 동일 철학).
런타임 노출 여부는 소스로 확정 불가 → side_keys에 ctx_/gen_ 시리즈를 통째 보존해 스모크에서 눈으로 확인.
"""
import re
import time

import aiohttp

# 우리가 보는 카운터(우선순위 순). base 이름으로 매칭하고 _total 접미사를 자동 시도.
_CTX_KEYS = ("ctx_completed_requests", "ctx_total_requests")
_GEN_KEYS = ("gen_completed_requests", "gen_total_requests")
_WORKER_KEY = "trtllm_request_success"   # 폴백/교차검증(워커 엔드포인트에 존재)

# Prometheus 텍스트 한 줄: `name{labels} value`  (HELP/TYPE 주석은 별도로 거름)
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")


def parse_prometheus_text(text: str) -> dict:
    """Prometheus exposition 텍스트 → {metric_name: float}. 라벨은 제거하고 같은 이름은 합산.
    (`#`로 시작하는 HELP/TYPE는 무시. 파싱 불가 라인은 건너뜀.)"""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v   # 라벨셋이 여러 개면 합산(role별 분리는 이름 접두로 이미 됨)
    return out


def _pick(parsed: dict, keys) -> float | None:
    """keys 중 먼저 존재하는 시리즈 값을 반환. `_total` 접미사 유무 모두 시도. 없으면 None."""
    for k in keys:
        for cand in (k + "_total", k):
            if cand in parsed:
                return parsed[cand]
    return None


async def snapshot(session: aiohttp.ClientSession, base_url: str) -> dict:
    """orchestrator(:8000) /prometheus/metrics를 1회 스냅샷.

    반환: {ts, ctx, gen, worker_success, side_keys}
      - ts            : wall-clock(초). 윈도우 폭 계산용.
      - ctx / gen     : prefill / decode 누적 완료수(없으면 None → analyze가 'n/a').
      - worker_success: trtllm_request_success_total (orchestrator엔 보통 없음; 폴백 진단용).
      - side_keys     : ctx_/gen_ 로 시작하는 모든 시리즈(스모크에서 실재/이름 확인용).
    실패(HTTP≠200/연결거부)면 ctx=gen=None + error — 측정을 막지 않음."""
    ts = time.time()
    try:
        async with session.get(
            f"{base_url}/prometheus/metrics",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return {"ts": ts, "ctx": None, "gen": None, "error": f"http_{resp.status}"}
            text = await resp.text()
    except Exception as exc:
        return {"ts": ts, "ctx": None, "gen": None, "error": str(exc)[:200]}

    parsed = parse_prometheus_text(text)
    return {
        "ts": ts,
        "ctx": _pick(parsed, _CTX_KEYS),
        "gen": _pick(parsed, _GEN_KEYS),
        "worker_success": parsed.get(_WORKER_KEY + "_total"),
        "side_keys": {k: v for k, v in parsed.items() if k.startswith(("ctx_", "gen_"))},
    }
