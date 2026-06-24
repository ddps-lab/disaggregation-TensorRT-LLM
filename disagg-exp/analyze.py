#!/usr/bin/env python3
"""
Post-experiment analysis for disagg-exp.

Usage:
    python analyze.py --log-dir ./results [--configs T1 T2 ...] [--plot]

각 포인트의 메트릭은 **공식 benchmark_serving 결과(`bench_<point>.json`)**에서 읽고,
per-side(prefill/decode) RPS·TPS는 `prom_<point>.json`, KV전송시간은 `perf_<point>.json`에서 병합한다.
(전체 TTFT/TPOT/ITL/E2EL/throughput = 공식 집계 그대로. 우리가 다시 계산하지 않음.)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPLOT = True
except ImportError:
    HAS_MPLOT = False

# 비용($/Mtok)은 analyze에서 계산하지 않는다 (사용자 결정 2026-06-24).
#   이유: $/Mtok = (인스턴스수 × $/hr) / output_throughput — throughput만 있으면 언제든
#   재계산되는 산수이고, 단가를 코드에 박으면 region/spot/시점에 따라 stale해짐.
#   → analyze는 측정치(throughput/latency/per-side)만 출력. 비용은 writeup에서 그 시점
#     단가를 측정 throughput에 곱해 외부 계산한다.


def _p(arr: list[float], pct: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(arr, pct))


def _num(v) -> float:
    """None/비수치 → NaN (플롯·포맷 안전)."""
    return float(v) if isinstance(v, (int, float)) and v == v else float("nan")


def _fmt(v, width: int, prec: int = 1) -> str:
    """수치면 폭/정밀도 맞춰 포맷, 아니면 'n/a'."""
    return f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) and v == v else f"{'n/a':>{width}}"


def parse_point_id(point_id: str) -> tuple[int, int, float]:
    """p{prefill}_d{decode}_r{rate} → (prefill, decode, rate)."""
    parts = point_id.split("_")
    return int(parts[0][1:]), int(parts[1][1:]), float(parts[2][1:])


def list_points(config_dir: Path) -> list[str]:
    """config 디렉토리의 bench_<point>.json들에서 point_id 목록 추출."""
    if not config_dir.exists():
        return []
    return sorted(p.stem[len("bench_"):] for p in config_dir.glob("bench_*.json"))


def load_bench(config_dir: Path, point_id: str) -> dict:
    """bench_{point_id}.json (공식 benchmark_serving --save-result)에서 집계 메트릭을 읽음.
    percentile 키는 sweep가 넘긴 --metric-percentiles(50,99) 기준(p50_*_ms / p99_*_ms).
    파일 없음/깨짐/실패면 {'n_ok':0} → 표에 NO DATA."""
    bf = config_dir / f"bench_{point_id}.json"
    if not bf.exists():
        return {"n_ok": 0}
    try:
        d = json.loads(bf.read_text())
    except Exception:
        return {"n_ok": 0}
    if not isinstance(d, dict):
        return {"n_ok": 0}
    completed = d.get("completed", 0) or 0
    num_prompts = d.get("num_prompts", 0) or 0
    fail_rate = (1.0 - completed / num_prompts) if num_prompts else float("nan")
    in_lens = d.get("input_lens") or []     # --save-detailed일 때만 존재
    out_lens = d.get("output_lens") or []
    mean_in = (sum(in_lens) / len(in_lens)) if in_lens else None
    mean_out = (sum(out_lens) / len(out_lens)) if out_lens else None
    return {
        "n_ok": completed,
        "num_prompts": num_prompts,
        "fail_rate": fail_rate,
        "fail_pct": fail_rate * 100,
        "ttft_p50_ms": d.get("p50_ttft_ms"),
        "ttft_p99_ms": d.get("p99_ttft_ms"),
        "tpot_p50_ms": d.get("p50_tpot_ms"),
        "tpot_p99_ms": d.get("p99_tpot_ms"),
        "itl_p99_ms":  d.get("p99_itl_ms"),
        "e2el_p99_ms": d.get("p99_e2el_ms"),
        "out_tok_s":   d.get("output_throughput"),
        "req_s":       d.get("request_throughput"),
        "total_tok_s": d.get("total_token_throughput"),
        "mean_input_len":  mean_in,
        "mean_output_len": mean_out,
    }


def _warn_pt_delta(point_id: str, bench: dict) -> None:
    """benchmark_serving이 실측한 평균 입력길이가 목표 prefill_len과 >1 어긋나면 경고(--save-detailed 필요)."""
    mean_in = bench.get("mean_input_len")
    if mean_in is None:
        return
    pl, _, _ = parse_point_id(point_id)
    if abs(mean_in - pl) > 1:
        print(f"  WARN [{point_id}]: mean input_len {mean_in:.1f} != prefill_len {pl}")


def load_perf(config_dir: Path, point_id: str) -> dict:
    """perf_{point_id}.json (orchestrator /perf_metrics 스냅샷)에서 KV전송시간·블록재사용 통계 추출.
    파일 없으면(=perf 비활성) 빈 dict → 표에 'n/a'. gen kv_cache_transfer_*는 kv_cache_size>0일 때만 존재."""
    pf = config_dir / f"perf_{point_id}.json"
    if not pf.exists():
        return {}
    try:
        data = json.loads(pf.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):   # 비활성/에러 응답(dict/null 등)이면 안전 skip (analyze 전체 크래시 방지)
        return {}
    kv_ms: list[float] = []
    reused = missed = 0
    for e in data:
        if not isinstance(e, dict):
            continue
        gen = e.get("gen_perf_metrics") or {}
        # 워커 per-request dict는 timing_metrics를 'perf_metrics' 래퍼 아래 중첩할 수 있음(버전차) → 둘 다 대응
        gen = gen.get("perf_metrics") or gen
        tm = gen.get("timing_metrics") or {}
        st, en = tm.get("kv_cache_transfer_start"), tm.get("kv_cache_transfer_end")
        if st is not None and en is not None:
            kv_ms.append((en - st) * 1000.0)
        km = gen.get("kv_cache_metrics") or {}
        reused += km.get("num_reused_blocks", 0)
        missed += km.get("num_missed_blocks", 0)
    out: dict = {"n_kv": len(kv_ms)}
    if kv_ms:
        out["kv_transfer_p50_ms"] = _p(kv_ms, 50)
        out["kv_transfer_p99_ms"] = _p(kv_ms, 99)
    if reused + missed:
        out["block_reuse_ratio"] = reused / (reused + missed)
    return out


def load_prom(config_dir: Path, point_id: str, mean_pt: float | None, mean_ct: float | None) -> dict:
    """prom_{point_id}.json (measured-윈도우 전/후 스냅샷)에서 per-side RPS를 계산하고,
    강제된 토큰길이(mean_pt/mean_ct)로 per-side TPS를 파생.
    - prefill_rps = (ctx_after − ctx_before)/window_s, decode_rps = (gen_after − gen_before)/window_s
      (ctx_/gen_completed_requests_total — prom_scrape.py 참조)
    - per-side TPS는 공식 토큰 카운터가 없어(LEARNING_NOTES.md §병렬화·KV전송·per-side 6번) RPS × 토큰수로 파생.
    파일 없으면(=스크레이프 비활성/orchestrator 미노출) 빈 dict → 표에 'n/a'."""
    pf = config_dir / f"prom_{point_id}.json"
    if not pf.exists():
        return {}
    try:
        snap = json.loads(pf.read_text())
    except Exception:
        return {}
    if not isinstance(snap, dict):
        return {}
    before = snap.get("before") or {}
    after = snap.get("after") or {}
    window_s = snap.get("window_s")
    if not window_s or window_s <= 0:                       # window_s 누락 시 ts로 폴백
        window_s = (after.get("ts") or 0) - (before.get("ts") or 0)
    if not window_s or window_s <= 0:
        return {}

    def _rps(side: str):
        b, a = before.get(side), after.get(side)
        if b is None or a is None:
            return None
        d = a - b
        return d / window_s if d >= 0 else None             # 카운터 리셋 등 음수면 무효

    pf_rps, dc_rps = _rps("ctx"), _rps("gen")

    out: dict = {}
    if pf_rps is not None:
        out["prefill_rps"] = pf_rps
        if mean_pt:
            out["prefill_tps"] = pf_rps * mean_pt
    if dc_rps is not None:
        out["decode_rps"] = dc_rps
        if mean_ct:
            out["decode_tps"] = dc_rps * mean_ct
    return out


def load_batch(config_dir: Path, point_id: str) -> dict:
    """batch_<point>.json (워커 /metrics 샘플) → per-side 배치수·KV사용 (mean). 없으면 {}.
    pf_bsz=prefill 평균 배치(numContextRequests), dc_bsz=decode 평균 배치(numGenRequests),
    dc_kv_pct=decode KV풀 사용률(usedNumBlocks/maxNumBlocks)."""
    p = config_dir / f"batch_{point_id}.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return {}
    out: dict = {}
    # pf_bsz/dc_bsz = "처리 중일 때"의 평균 배치수(idle 0 제외) — 먼저 끝난 쪽이 0으로 평균 깎는 것 방지.
    #   (전체평균 prefill_batch_mean·바쁜비율 *_active_frac·최대 *_batch_max 는 batch_<point>.json에 그대로 있음)
    pf = d.get("prefill_batch_mean_active", d.get("prefill_batch_mean"))
    if pf is not None:
        out["pf_bsz"] = pf
    dc = d.get("decode_batch_mean_active", d.get("decode_batch_mean"))
    if dc is not None:
        out["dc_bsz"] = dc
    if d.get("decode_kv_used_frac_mean") is not None:
        out["dc_kv_pct"] = d["decode_kv_used_frac_mean"] * 100
    # backlog = (ctx_completed − gen_completed)의 윈도우 평균/최대 = "prefill은 끝났는데 decode는
    #   아직 안 끝난 요청 수" = prefill 산출이 decode 앞에 쌓인 양. 관측값만 기록(해석은 사용자).
    if d.get("backlog_mean") is not None:
        out["backlog"] = d["backlog_mean"]
    if d.get("backlog_max") is not None:
        out["backlog_max"] = d["backlog_max"]
    # 바쁜 시간 비율(%) — prefill이 "끝내고 노는" 정도가 여기 보임(낮을수록 idle 많음)
    if d.get("prefill_active_frac") is not None:
        out["pf_act"] = d["prefill_active_frac"] * 100
    if d.get("decode_active_frac") is not None:
        out["dc_act"] = d["decode_active_frac"] * 100
    return out


# 표/CSV 공통 컬럼 스펙: (표시이름, stats키, 너비, 소수자리). 카테고리별 4개 서브-표.
_TABLE_GROUPS = [
    ("① 지연 latency (ms)", [
        ("n_ok", "n_ok", 5, 0), ("fail%", "fail_pct", 6, 1),
        ("ttft_p50", "ttft_p50_ms", 9, 1), ("ttft_p99", "ttft_p99_ms", 9, 1),
        ("tpot_p50", "tpot_p50_ms", 9, 2), ("tpot_p99", "tpot_p99_ms", 9, 2),
        ("itl_p99", "itl_p99_ms", 8, 2), ("e2el_p99", "e2el_p99_ms", 9, 1),
    ]),
    ("② 처리량 throughput + KV전송(ms)", [
        ("out_tok/s", "out_tok_s", 10, 1),
        ("kv_p50", "kv_transfer_p50_ms", 8, 2), ("kv_p99", "kv_transfer_p99_ms", 8, 2),
    ]),
    ("③ per-side 완료율·토큰 (prefill vs decode)", [
        ("pf_rps", "prefill_rps", 8, 2), ("dc_rps", "decode_rps", 8, 2),
        ("pf_tps", "prefill_tps", 8, 0), ("dc_tps", "decode_tps", 8, 0),
    ]),
    ("④ per-side 배치·점유·backlog (배치=처리중 평균, act%=바쁜시간, backlog=쌓인수)", [
        ("pf_bsz", "pf_bsz", 7, 2), ("dc_bsz", "dc_bsz", 7, 2),
        ("pf_act%", "pf_act", 8, 1), ("dc_act%", "dc_act", 8, 1),
        ("dc_kv%", "dc_kv_pct", 7, 1), ("backlog", "backlog", 8, 1),
    ]),
]


def format_tables(all_stats: dict[str, dict[str, dict]]) -> str:
    """4개 서브-표를 문자열로 렌더(터미널 출력 + 파일 저장 공용). 행 키 = (config, point)."""
    keys = [(c, p) for c in sorted(all_stats) for p in sorted(all_stats[c])]
    lines: list[str] = []
    for title, cols in _TABLE_GROUPS:
        hdr = f"{'config':<8} {'point':<24}" + "".join(f" {n:>{w}}" for n, _, w, _ in cols)
        lines += [f"\n── {title} ──", hdr, "-" * len(hdr)]
        for c, p in keys:
            s = all_stats[c][p]
            row = f"{c:<8} {p:<24}"
            for _, key, w, prec in cols:
                row += f" {_fmt(s.get(key), w, prec)}"
            lines.append(row)
    return "\n".join(lines)


def write_csv(all_stats: dict[str, dict[str, dict]], path: Path) -> None:
    """표와 동일 컬럼을 한 줄=한 포인트로 CSV 저장(논문용, RAW 수치 — 반올림/포맷 안 함)."""
    cols = [("config", None), ("point", None)]
    for _, group_cols in _TABLE_GROUPS:
        cols += [(name, key) for name, key, _, _ in group_cols]
    keys = [(c, p) for c in sorted(all_stats) for p in sorted(all_stats[c])]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([n for n, _ in cols])
        for c, p in keys:
            s = all_stats[c][p]
            w.writerow([
                (c if name == "config" else p) if key is None
                else ("" if s.get(key) is None else s.get(key))
                for name, key in cols
            ])


def plot_comparison(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """One plot per (prefill_len, decode_len) pair, 4 panels vs rate, all configs:
    TTFT / per-side 완료 rps(prefill vs decode) / output throughput / batch & backlog."""
    if not HAS_MPLOT:
        print("matplotlib not available, skipping plots")
        return

    by_pd: dict[tuple, dict[str, list]] = defaultdict(dict)
    for config, points in all_stats.items():
        for point_id, s in points.items():
            try:
                pl, dl, r = parse_point_id(point_id)
            except Exception:
                continue
            by_pd[(pl, dl)].setdefault(config, []).append((r, s))

    for (pl, dl), config_data in by_pd.items():
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f"prefill={pl} decode={dl}")

        for config, rate_stats in sorted(config_data.items()):
            rate_stats.sort(key=lambda x: x[0])
            rates  = [x[0] for x in rate_stats]
            ttft50 = [_num(x[1].get("ttft_p50_ms")) for x in rate_stats]
            ttft99 = [_num(x[1].get("ttft_p99_ms")) for x in rate_stats]
            thr     = [_num(x[1].get("out_tok_s")) for x in rate_stats]
            pf_rps  = [_num(x[1].get("prefill_rps")) for x in rate_stats]
            dc_rps  = [_num(x[1].get("decode_rps")) for x in rate_stats]
            pf_bsz  = [_num(x[1].get("pf_bsz")) for x in rate_stats]
            dc_bsz  = [_num(x[1].get("dc_bsz")) for x in rate_stats]
            backlog = [_num(x[1].get("backlog")) for x in rate_stats]

            axes[0].plot(rates, ttft50, marker="o", label=f"{config} p50")
            axes[0].plot(rates, ttft99, marker="x", linestyle="--", label=f"{config} p99")
            axes[1].plot(rates, pf_rps, marker="o", label=f"{config} prefill")
            axes[1].plot(rates, dc_rps, marker="s", linestyle="--", label=f"{config} decode")
            axes[2].plot(rates, thr, marker="o", label=config)
            axes[3].plot(rates, pf_bsz, marker="o", label=f"{config} pf_bsz")
            axes[3].plot(rates, dc_bsz, marker="s", linestyle="--", label=f"{config} dc_bsz")
            axes[3].plot(rates, backlog, marker="^", linestyle=":", label=f"{config} backlog")

        axes[0].set_title("TTFT (ms)");        axes[0].set_xlabel("rate (req/s)"); axes[0].legend(fontsize=7)
        axes[1].set_title("per-side 완료 rps (prefill vs decode)"); axes[1].set_xlabel("rate (req/s)"); axes[1].legend(fontsize=7)
        axes[2].set_title("output throughput (tok/s)"); axes[2].set_xlabel("rate (req/s)"); axes[2].legend(fontsize=7)
        axes[3].set_title("batch & backlog (req)"); axes[3].set_xlabel("rate (req/s)"); axes[3].legend(fontsize=7)

        fig.tight_layout()
        fname = out_dir / f"plot_p{pl}_d{dl}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  saved {fname}")


def main(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    configs = args.configs or ["T1", "T2", "T3", "T4"]

    all_stats: dict[str, dict[str, dict]] = {}
    for config in configs:
        config_dir = log_dir / config
        point_ids = list_points(config_dir)
        if not point_ids:
            print(f"[analyze] no data for config {config} in {config_dir}")
            continue
        all_stats[config] = {}
        for point_id in point_ids:
            b = load_bench(config_dir, point_id)
            _warn_pt_delta(point_id, b)
            pl, dl, _ = parse_point_id(point_id)
            mean_pt = b.get("mean_input_len") or pl     # --save-detailed 없으면 목표 길이로 폴백
            mean_ct = b.get("mean_output_len") or dl
            s = dict(b)
            s.update(load_perf(config_dir, point_id))                       # KV전송시간 (있으면)
            s.update(load_prom(config_dir, point_id, mean_pt, mean_ct))     # per-side RPS/TPS (있으면)
            s.update(load_batch(config_dir, point_id))                      # per-side 배치수·KV사용 (있으면)
            all_stats[config][point_id] = s

    if not all_stats:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    txt = format_tables(all_stats)
    print(txt)

    # 터미널뿐 아니라 파일로도 저장 (스크롤로 날아가지 않게 + 논문용 RAW CSV).
    tag = "_".join(sorted(all_stats))
    summary_txt = log_dir / f"summary_{tag}.txt"
    summary_csv = log_dir / f"summary_{tag}.csv"
    summary_txt.write_text(txt + "\n")
    write_csv(all_stats, summary_csv)
    print(f"\n[analyze] 표 저장 → {summary_txt}")
    print(f"[analyze] CSV 저장 → {summary_csv}")

    if args.plot:
        out_dir = log_dir / "plots"
        out_dir.mkdir(exist_ok=True)
        plot_comparison(all_stats, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=os.environ.get("EXP_LOG_DIR", "./results"))
    parser.add_argument("--configs", nargs="*", help="Which configs to analyze (default: all found)")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figures")
    args = parser.parse_args()
    main(args)
