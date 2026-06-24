"""metrics_sampler.py — per-side(prefill/decode) 배치수·KV사용 기록 (공식 미제공 → 커스텀).

orchestrator(:8000)는 per-side "지금 돌고 있는 요청수(배치수)"를 집계해 주지 않는다.
그 값은 각 워커의 GET /metrics (iteration stats)에만 라이브로 노출되므로, measured 윈도우
동안 워커 /metrics를 주기 폴링해 prefill/decode 배치수와 KV 사용량의 평균/최대를 낸다.

스키마는 v1.2.1 소스로 검증함 (tests/unittest/llmapi/apps/_test_openai_metrics.py:54-101):
  - /metrics 는 stat 객체 "리스트"(최근 iteration들)를 반환
  - stat["inflightBatchingStats"]["numContextRequests"]  # prefill(context) 배치
  - stat["inflightBatchingStats"]["numGenRequests"]       # decode(generation) 배치
  - stat["kvCacheStats"]["usedNumBlocks"] / ["maxNumBlocks"] / ["tokensPerBlock"]
camelCase. PyTorch 백엔드도 동일 스키마(같은 IterationStats 직렬화).

⚠️ inter-node에선 gen 워커가 원격 노드라, sweep 명령에 CTX_URLS/GEN_URLS를 줘서
   원격 워커 /metrics 주소를 알려줘야 한다(orchestrator엔 iteration stats 없음).
"""
import asyncio
from collections import defaultdict

import aiohttp

# side → 그 워커의 "현재 배치수"를 나타내는 inflightBatchingStats 필드
_BATCH_FIELD = {"prefill": "numContextRequests", "decode": "numGenRequests"}


class BatchSampler:
    """measured 윈도우 동안 워커 /metrics를 interval초마다 폴링해 per-side 배치수·KV사용 누적.

    endpoints: [(side, base_url)], side ∈ {"prefill","decode"}, base_url 예 "http://172.31.50.92:8011"
    """

    def __init__(self, session: aiohttp.ClientSession,
                 endpoints, interval: float = 1.0):
        self._session = session
        self._endpoints = list(endpoints)
        self._interval = interval
        self._stop = asyncio.Event()
        self._task = None
        self._samples = defaultdict(list)   # side → [{"batch":int, "used":int, "maxb":int}]

    async def _poll_one(self, side: str, base_url: str) -> None:
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

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.gather(*[self._poll_one(s, u) for s, u in self._endpoints])
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
                out[f"{side}_batch_mean"] = sum(batches) / len(batches)
                out[f"{side}_batch_max"] = max(batches)
            if fracs:
                out[f"{side}_kv_used_frac_mean"] = sum(fracs) / len(fracs)
                out[f"{side}_kv_used_frac_max"] = max(fracs)
        return out
