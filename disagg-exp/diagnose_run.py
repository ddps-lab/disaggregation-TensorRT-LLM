#!/usr/bin/env python3
"""diagnose_run.py — KV풀 backpressure/크래시 원인 자동 추출 (RCA 재실험용).

자고 일어나 재실험한 뒤 이 스크립트 하나로 Q1·Q2 판정에 필요한 증거를 워커 로그 +
sampler trace에서 뽑는다. 추측 안 하고 "로그에 있는 사실"만 출력한다.

판정에 쓰는 신호(전부 프레임워크가 debug 로그/메트릭으로 이미 내보냄, 켜기만 하면 됨):
  Q1 (prefill이 ctx풀 꽉 차서 큐에서 admit 못 함):
    - timeseries trace의 backlog 평탄 구간에서 prefill_kv_frac≈1.0 AND prefill_queue>0 인가?
    - gen 로그에 "may not have enough kvCache" 경고(py_executor.py:1037)가 그 시점에 뜨는가?
  Q2 (p128_d2048 크래시 원인):
    - gen 로그에 assert "Can't allocate new blocks. No free blocks left." 가 있나?
    - 있으면 그 발생 파일:라인(kvCacheManager.cpp:1584 receive-add_sequence vs :1508 per-step addToken
      vs :1732 replaceSharedBlock) — 어느 호출부인지 = 진짜 메커니즘.
    - 시작 로그의 capacity_scheduler_policy = GUARANTEED_NO_EVICT 확인(풀-출력 예약 ⇒ over-admit 아님 검증).

사용:
  python3 diagnose_run.py --log-dir <run의 LOG_DIR>          # 워커 로그 + trace 자동탐색
  python3 diagnose_run.py --log-dir results/T1               # analyze 결과 폴더(raw/ 안 trace) 도 가능
경로 가정(launch_trtllm.sh): 워커 로그 = <LOG_DIR>/logs/trtllm_*_{ctx,gen,context,generation}_*.log,
  trace = <LOG_DIR>/.../timeseries_<point>.jsonl (raw/ 또는 평면).
"""
import argparse
import glob
import json
import re
from pathlib import Path

# 워커 로그에서 찾을 패턴(verbatim, 소스 기준)
PATTERNS = {
    "warn_not_enough_kv": re.compile(r"may not have enough kvCache"),
    "crash_no_free_blocks": re.compile(r"Can'?t allocate new blocks\. No free blocks left"),
    "policy": re.compile(r"capacity_scheduler_policy['\"=:\s]+([A-Z_]+)", re.I),
    "kvmgr_origin": re.compile(r"kvCacheManager\.cpp:(\d+)"),
    "maxblocks": re.compile(r"max(?:Num)?Blocks['\"=:\s]+(\d+)", re.I),
    "tokens_per_block": re.compile(r"tokens[_ ]?per[_ ]?block['\"=:\s]+(\d+)", re.I),
}
ORIGIN_MEANING = {
    "1584": "receive-path add_sequence (전송 받을 때 할당) — 예약 밖 할당 의심",
    "1508": "per-step addToken growth (decode 토큰 자랄 때) — 예약 부족 의심",
    "1509": "per-step addToken growth (decode 토큰 자랄 때) — 예약 부족 의심",
    "1510": "per-step addToken growth (decode 토큰 자랄 때) — 예약 부족 의심",
    "1511": "per-step addToken growth (decode 토큰 자랄 때) — 예약 부족 의심",
    "1732": "replaceSharedBlock (블록 공유 교체) — block_reuse 경로",
}


def find_logs(log_dir: Path):
    """ctx/gen 워커 로그 파일 찾기(여러 네이밍 대응)."""
    cands = []
    for base in (log_dir, log_dir / "logs", log_dir / "telemetry"):
        if base.exists():
            cands += glob.glob(str(base / "*.log"))
    ctx = [f for f in cands if re.search(r"(ctx|context)", f, re.I)]
    gen = [f for f in cands if re.search(r"(gen|generation)", f, re.I) and "ctx" not in f.lower()]
    other = [f for f in cands if f not in ctx and f not in gen]
    return {"ctx": sorted(set(ctx)), "gen": sorted(set(gen)), "other": sorted(set(other))}


def scan_log(path: str) -> dict:
    """한 워커 로그에서 패턴별 카운트/첫발생/발생라인 추출."""
    out = {"file": path, "warn": 0, "crash": 0, "policy": None,
           "crash_origins": {}, "maxblocks": None, "tokens_per_block": None,
           "first_warn_line": None, "crash_lines": []}
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except Exception as e:
        out["error"] = str(e)
        return out
    for i, ln in enumerate(lines):
        if PATTERNS["warn_not_enough_kv"].search(ln):
            out["warn"] += 1
            if out["first_warn_line"] is None:
                out["first_warn_line"] = ln.strip()[:200]
        if PATTERNS["crash_no_free_blocks"].search(ln):
            out["crash"] += 1
            # 발생 호출부: 같은 줄 또는 근처 ±3줄에서 kvCacheManager.cpp:NNNN
            ctx_win = "\n".join(lines[max(0, i - 3):i + 4])
            for m in PATTERNS["kvmgr_origin"].finditer(ctx_win):
                out["crash_origins"][m.group(1)] = out["crash_origins"].get(m.group(1), 0) + 1
            out["crash_lines"].append(ln.strip()[:200])
        if out["policy"] is None:
            mp = PATTERNS["policy"].search(ln)
            if mp:
                out["policy"] = mp.group(1)
        if out["maxblocks"] is None:
            mb = PATTERNS["maxblocks"].search(ln)
            if mb:
                out["maxblocks"] = int(mb.group(1))
        if out["tokens_per_block"] is None:
            tb = PATTERNS["tokens_per_block"].search(ln)
            if tb:
                out["tokens_per_block"] = int(tb.group(1))
    return out


def analyze_traces(log_dir: Path) -> list:
    """timeseries_<point>.jsonl 에서 Q1 결정규칙 신호 추출(신규 키 prefill_kv_frac 필요)."""
    files = (glob.glob(str(log_dir / "**" / "timeseries_*.jsonl"), recursive=True)
             or glob.glob(str(log_dir / "timeseries_*.jsonl")))
    rows_out = []
    for f in sorted(set(files)):
        pt = Path(f).name.replace("timeseries_", "").replace(".jsonl", "")
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        if not rows:
            continue
        bl = [(r.get("backlog") or 0) for r in rows]
        if not any(bl):
            continue
        peak = max(bl)
        # 평탄(plateau) 구간 = backlog ≥ 0.8×peak 인 샘플들
        plateau = [r for r in rows if (r.get("backlog") or 0) >= 0.8 * peak]
        def avg(key):
            vs = [r.get(key) for r in plateau if isinstance(r.get(key), (int, float))]
            return round(sum(vs) / len(vs), 3) if vs else None
        has_ctxkv = any(r.get("prefill_kv_frac") is not None for r in rows)
        has_q = any(r.get("prefill_queue") is not None for r in rows)
        rows_out.append({
            "point": pt, "n_ticks": len(rows), "backlog_peak": peak,
            "plateau_ticks": len(plateau),
            "ctx_kv_frac@plateau": avg("prefill_kv_frac"),     # ← Q1 핵심(없으면 None=재런필요)
            "decode_kv_frac@plateau": avg("decode_kv_frac"),
            "prefill_queue@plateau": avg("prefill_queue"),     # ← Q1 핵심(>0이어야 backpressure)
            "prefill_batch@plateau": avg("prefill_batch"),
            "_has_ctx_kv": has_ctxkv, "_has_prefill_queue": has_q,
        })
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="재실험 LOG_DIR (워커 로그 + trace 위치)")
    a = ap.parse_args()
    log_dir = Path(a.log_dir)

    print(f"# diagnose_run — {log_dir}\n")
    logs = find_logs(log_dir)
    print("## 워커 로그")
    if not any(logs.values()):
        print("  (로그 없음 — LOG_LEVEL=debug로 재실험했는지, 경로 맞는지 확인)\n")
    scans = {}
    for role in ("ctx", "gen", "other"):
        for f in logs[role]:
            s = scan_log(f)
            scans[f] = (role, s)
            origins = ", ".join(f"kvCacheManager.cpp:{k}×{v} ({ORIGIN_MEANING.get(k,'?')})"
                                for k, v in s["crash_origins"].items()) or "-"
            print(f"  [{role}] {Path(f).name}")
            print(f"     policy={s.get('policy')}  maxBlocks={s.get('maxblocks')}  tok/blk={s.get('tokens_per_block')}")
            print(f"     'not enough kvCache' 경고={s['warn']}회  |  크래시 assert={s['crash']}회")
            if s["crash"]:
                print(f"     ★ 크래시 발생부: {origins}")
            if s.get("first_warn_line"):
                print(f"     첫 경고: {s['first_warn_line']}")

    print("\n## Q2 판정 (크래시 원인)")
    crashes = {f: s for f, (r, s) in scans.items() if s["crash"]}
    policies = {s.get("policy") for _, s in scans.values() if s.get("policy")}
    if not crashes:
        print("  크래시 assert 없음 → 이 빌드/설정선 안 멈춤(또는 로그에 안 잡힘). 원래 크래시는 다른 config였을 수 있음.")
    else:
        allo = {}
        for s in crashes.values():
            for k, v in s["crash_origins"].items():
                allo[k] = allo.get(k, 0) + v
        print(f"  크래시 발생부 집계: {allo or '(라인 못 찾음 — DEBUG/stack 필요)'}")
        for k in allo:
            print(f"    → kvCacheManager.cpp:{k} = {ORIGIN_MEANING.get(k,'알수없는 호출부, 소스 확인')}")
    if policies:
        print(f"  gen capacity_scheduler_policy = {policies}  (GUARANTEED_NO_EVICT면 풀-출력 예약 ⇒ over-admit설 거짓)")

    print("\n## Q1 판정 (prefill backpressure)")
    traces = analyze_traces(log_dir)
    if not traces:
        print("  timeseries trace 없음.")
    for t in traces:
        if not t["_has_ctx_kv"]:
            print(f"  {t['point']}: ctx_kv 없음 → 이번 trace는 prefill_kv_frac 미기록(구 sampler). 재런 필요.")
            continue
        ckv, pq = t["ctx_kv_frac@plateau"], t["prefill_queue@plateau"]
        verdict = "?"
        if ckv is not None and pq is not None:
            if ckv >= 0.97 and pq > 0:
                verdict = "✅ CONFIRMED — ctx풀 거의 꽉참 + 큐 대기>0 = prefill이 ctx-KV로 admit 막힘"
            elif ckv < 0.9:
                verdict = "❌ FALSIFIED — ctx풀 안 참(다른 throttle). backlog는 단순 rate-matching"
            else:
                verdict = "△ 애매 — ctx_kv 중간. 경고로그/샘플레이트 같이 봐야"
        print(f"  {t['point']}: ctx_kv@plateau={ckv}  prefill_queue@plateau={pq}  prefill_batch={t['prefill_batch@plateau']}  → {verdict}")

    print("\n(판정 규칙·재실험 셋업 = RCA_kvpool_reexperiment.md)")


if __name__ == "__main__":
    main()
