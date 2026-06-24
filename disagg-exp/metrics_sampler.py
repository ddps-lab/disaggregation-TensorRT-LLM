"""metrics_sampler.py — per-side(prefill/decode) 라이브 상태 + backlog 기록 (공식 미제공 → 커스텀).

measured 윈도우 동안 1Hz로 폴링해, result(논문)에 들어갈 per-side 동역학을 그대로 기록한다.
추천/비율 판단은 하지 않는다 — 관측값만 충실히 남기고 해석은 사용자가 한다.
  - prefill/decode 동시 배치수 : 각 워커 /metrics inflightBatchingStats.numContext/numGenRequests
  - decode KV풀 사용률          : 워커 /metrics kvCacheStats.used/maxNumBlocks
  - backlog(쌓인 요청수)        : orchestrator /prometheus/metrics 의 ctx_completed − gen_completed
       = "prefill은 끝났는데 decode는 아직 안 끝난 요청 수" (openai_client.py:266-267 _finish_request에서
         ctx/gen 클라이언트가 각자 완료 때 .inc; perf_metrics.py:93-99 role 접두 ctx/gen).
또한 SWEEP_LIVE면 진행 중 per-side 상태를 stderr로 주기 출력(상태줄).

스키마 검증(v1.2.1): tests/unittest/llmapi/apps/_test_openai_metrics.py:54-101
  (inflightBatchingStats / kvCacheStats, camelCase, /metrics = stat 리스트). PyTorch 백엔드 동일.
"""
import asyncio
import sys
import time
from collections import defaultdict

import aiohttp

import prom_scrape   # orchestrator ctx/gen 완료 카운터 스냅샷 재사용

# side → 그 워커의 "현재 배치수"를 나타내는 inflightBatchingStats 필드
_BATCH_FIELD = {"prefill": "numContextRequests", "decode": "numGenRequests"}


class BatchSampler:
    """measured 윈도우 동안 워커 /metrics + orchestrator 카운터를 주기 폴링.

    endpoints: [(side, base_url)], side ∈ {"prefill","decode"}, base_url 예 "http://172.31.50.92:8011"
    orchestrator_url: backlog(ctx-gen)용 orchestrator base_url (예 "http://localhost:8000")
    """

    def __init__(self, session: aiohttp.ClientSession, endpoints, interval: float = 1.0,
                 orchestrator_url=None, live: bool = True, live_every: int = 5):
        self._session = session
        self._endpoints = list(endpoints)
        self._interval = interval
        self._orch = orchestrator_url
        self._live = live
        self._live_every = max(1, live_every)
        self._stop = asyncio.Event()
        self._task = None
        self._samples = defaultdict(list)   # side → [{batch, used, maxb}]
        self._backlog = []                  # (ctx_completed - gen_completed) 샘플 = 쌓인 수
        self._last_ctx = None
        self._last_gen = None
        self._last_ctx_total = None          # ctx_total_requests_total(=context에 들어온 누적) → prefill 큐 계산용
        self._ctx0 = None                   # 측정 시작 시점 누적값(baseline) → 완료수를 이 실험 기준 0부터로
        self._gen0 = None
        self._tick = 0
        self._t_start = None                # 실제 경과시간(t축)용 — tick×interval 대신 monotonic 기준
        self.trace = []                     # 매 tick 스냅샷(라이브 기록) → live_<point>.jsonl로 저장

    async def _poll_worker(self, side: str, base_url: str) -> None:
        # ⚠️ 워커 /metrics(get_iteration_stats)는 내부적으로 get_stats_async(timeout=2)로
        #   stats 큐를 "최대 2초" 기다렸다 응답(llm.py:585; idle이면 꽉 2초 블록). 클라 타임아웃이
        #   2초면 레이스로 자주 끊겨 batch가 null이 됨 → 6초로 여유(서버 2초 + 마진).
        try:
            async with self._session.get(
                    f"{base_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=6)) as r:
                if r.status != 200:
                    return
                data = await r.json()
        except Exception:
            return  # 폴링 실패는 조용히 무시(측정 자체엔 영향 없음)
        if not isinstance(data, list):
            return
        field = _BATCH_FIELD[side]
        for stat in data:
            ibs = stat.get("inflightBatchingStats") or {}
            kvs = stat.get("kvCacheStats") or {}
            self._samples[side].append({
                "batch": ibs.get(field),
                "used": kvs.get("usedNumBlocks"),
                "maxb": kvs.get("maxNumBlocks"),
            })

    async def _poll_backlog(self) -> None:
        if not self._orch:
            return
        try:
            snap = await prom_scrape.snapshot(self._session, self._orch)
        except Exception:
            return
        ctx, gen = snap.get("ctx"), snap.get("gen")
        ctx_total = (snap.get("side_keys") or {}).get("ctx_total_requests_total")
        if isinstance(ctx, (int, float)) and isinstance(gen, (int, float)):
            if self._ctx0 is None:          # 첫 유효 샘플 = 이 실험의 baseline(완료수 0 기준점)
                self._ctx0, self._gen0 = ctx, gen
            self._backlog.append(ctx - gen)  # 차이라 baseline 무관(현재 outstanding)
            self._last_ctx, self._last_gen = ctx, gen
            if isinstance(ctx_total, (int, float)):
                self._last_ctx_total = ctx_total

    def _last_batch(self, side: str):
        rows = self._samples.get(side) or []
        return rows[-1]["batch"] if rows else None

    def _last_kv_frac(self, side: str):
        """그 측 KV풀 사용률(usedNumBlocks/maxNumBlocks, 0~1). 최근 유효 샘플. → 시계열 패널용."""
        for r in reversed(self._samples.get(side) or []):
            u, m = r.get("used"), r.get("maxb")
            if isinstance(u, (int, float)) and isinstance(m, (int, float)) and m:
                return u / m
        return None

    @staticmethod
    def _rel(val, base):
        """누적 카운터를 이 실험 시작(baseline) 기준 상대값으로. = 이 실험에서 완료한 수."""
        if isinstance(val, (int, float)) and isinstance(base, (int, float)):
            return val - base
        return None

    def _elapsed(self) -> float:
        return (time.monotonic() - self._t_start) if self._t_start is not None else 0.0

    def _prefill_queue(self):
        """context에 도착했지만 아직 prefill 안 끝난 수 = ctx_total − ctx_completed (순간값).
        이게 쌓이면 prefill이 도착을 못 따라가거나 backpressure로 막힌 것."""
        if isinstance(self._last_ctx_total, (int, float)) and isinstance(self._last_ctx, (int, float)):
            return self._last_ctx_total - self._last_ctx
        return None

    def _print_live(self) -> None:
        bl = self._backlog[-1] if self._backlog else None
        secs = self._elapsed()
        print(f"  [live +{secs:.0f}s] prefill_q={self._prefill_queue()} "
              f"prefill_done={self._rel(self._last_ctx, self._ctx0)} "
              f"decode_done={self._rel(self._last_gen, self._gen0)} "
              f"backlog={bl} | prefill_batch={self._last_batch('prefill')} decode_batch={self._last_batch('decode')}",
              file=sys.stderr, flush=True)

    async def _loop(self) -> None:
        self._t_start = time.monotonic()          # t축 기준점(실제 경과시간)
        while not self._stop.is_set():
            await asyncio.gather(*[self._poll_worker(s, u) for s, u in self._endpoints])
            await self._poll_backlog()
            self.trace.append({                       # 라이브 기록(매 tick) — timeseries_<point>.jsonl
                "t": round(self._elapsed(), 1),       # 실제 경과초(폴이 느려도 정확)
                # 완료수는 이 실험 시작 기준 0부터(누적 아님) — 실험별로 따로 보게.
                "prefill_done": self._rel(self._last_ctx, self._ctx0),
                "decode_done": self._rel(self._last_gen, self._gen0),
                "prefill_queue": self._prefill_queue(),                          # context 도착·prefill 대기 수(=ctx_total−ctx_completed)
                "backlog": (self._backlog[-1] if self._backlog else None),       # prefill끝·decode대기 수(buffer)
                "prefill_batch": self._last_batch("prefill"),                    # 그 순간 동시 배치수
                "decode_batch": self._last_batch("decode"),
                "decode_kv_frac": self._last_kv_frac("decode"),                  # decode KV풀 사용률 0~1(시계열 패널②)
            })
            if self._live and self._tick % self._live_every == 0:
                self._print_live()
            self._tick += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> dict:
        self._stop.set()
        if self._task is not None:
            await self._task
        return self._aggregate()

    def _aggregate(self) -> dict:
        # _legend: 이 파일의 키가 뭔지 자체 설명 ({side}=prefill|decode). 출처/식까지.
        out: dict = {"_legend": {
            "출처": "batch·kv = 워커 /metrics (inflightBatchingStats/kvCacheStats), "
                    "backlog = orchestrator ctx/gen_completed 카운터 차. 모두 1Hz 샘플.",
            "{side}_batch_mean": "동시 배치수 평균 (idle 0 포함 = 점유율 관점)",
            "{side}_batch_mean_active": "동시 배치수 평균 (idle 0 제외 = 처리 중일 때만). 표 prefill_batch/decode_batch가 이것",
            "{side}_batch_max": "관측된 최대 동시 배치수",
            "{side}_active_frac": "batch>0(바쁜) 샘플 비율 0~1 (표 *_busy% = ×100)",
            "{side}_kv_used_frac_mean/max": "KV풀 사용률 usedNumBlocks/maxNumBlocks 평균/최대 (표 decode_kv% = ×100)",
            "{side}_n_samples": "그 측면 1Hz 샘플 개수",
            "backlog_mean/max": "(ctx_completed - gen_completed) 평균/최대 = prefill 끝났는데 decode 아직 안 끝난 요청수",
            "backlog_n_samples": "backlog 샘플 개수",
        }}
        for side, rows in self._samples.items():
            batches = [r["batch"] for r in rows if isinstance(r["batch"], (int, float))]
            fracs = [r["used"] / r["maxb"] for r in rows
                     if isinstance(r["used"], (int, float))
                     and isinstance(r["maxb"], (int, float)) and r["maxb"]]
            out[f"{side}_n_samples"] = len(rows)
            if batches:
                active = [b for b in batches if b > 0]   # idle(0) 제외 — 먼저 끝난 쪽이 0으로 평균 깎는 것 방지
                out[f"{side}_batch_mean"] = sum(batches) / len(batches)            # 전체(idle 포함) = 점유율
                out[f"{side}_batch_mean_active"] = (sum(active) / len(active)) if active else 0.0  # 처리 중일 때만
                out[f"{side}_batch_max"] = max(batches)
                out[f"{side}_active_frac"] = len(active) / len(batches)            # 바쁜 시간 비율(0~1)
            if fracs:
                out[f"{side}_kv_used_frac_mean"] = sum(fracs) / len(fracs)
                out[f"{side}_kv_used_frac_max"] = max(fracs)
        if self._backlog:
            out["backlog_mean"] = sum(self._backlog) / len(self._backlog)
            out["backlog_max"] = max(self._backlog)
            out["backlog_n_samples"] = len(self._backlog)
        return out
