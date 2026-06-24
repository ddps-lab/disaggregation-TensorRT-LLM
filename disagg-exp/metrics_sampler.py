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
        self._tick = 0
        self.trace = []                     # 매 tick 스냅샷(라이브 기록) → live_<point>.jsonl로 저장

    async def _poll_worker(self, side: str, base_url: str) -> None:
        try:
            async with self._session.get(
                    f"{base_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=2)) as r:
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
        if isinstance(ctx, (int, float)) and isinstance(gen, (int, float)):
            self._backlog.append(ctx - gen)
            self._last_ctx, self._last_gen = ctx, gen

    def _last_batch(self, side: str):
        rows = self._samples.get(side) or []
        return rows[-1]["batch"] if rows else None

    def _print_live(self) -> None:
        bl = self._backlog[-1] if self._backlog else None
        secs = self._tick * self._interval
        print(f"  [live +{secs:.0f}s] ctx_done={self._last_ctx} gen_done={self._last_gen} "
              f"backlog={bl} | pf_bsz={self._last_batch('prefill')} dc_bsz={self._last_batch('decode')}",
              file=sys.stderr, flush=True)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.gather(*[self._poll_worker(s, u) for s, u in self._endpoints])
            await self._poll_backlog()
            self.trace.append({                       # 라이브 기록(매 tick) — 나중에 live_<point>.jsonl
                "t": round(self._tick * self._interval, 1),
                "ctx_done": self._last_ctx, "gen_done": self._last_gen,
                "backlog": (self._backlog[-1] if self._backlog else None),
                "pf_bsz": self._last_batch("prefill"), "dc_bsz": self._last_batch("decode"),
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
        out: dict = {}
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
