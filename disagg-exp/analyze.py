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

# AWS On-Demand $/hr — us-east-1 기준(2025). ⚠️ 리전·변동에 따라 반드시 재확인.
#   확인됨: g5.12xlarge=5.672, g6.xlarge=0.8048, g6e.xlarge=1.861
#          (출처 aws.amazon.com/ec2/pricing/on-demand)
#   근사(재확인 필요): g5.xlarge≈1.006, g6.12xlarge≈4.6016, g6e.12xlarge≈10.49
_INSTANCE_HR = {
    "g5.xlarge": 1.006,  "g5.12xlarge": 5.672,
    "g6.xlarge": 0.8048, "g6.12xlarge": 4.6016,
    "g6e.xlarge": 1.861, "g6e.12xlarge": 10.49,
}
# config 라벨별 총 $/hr = 그 config가 점유한 인스턴스 합.
# ⚠️ 키는 sweep.py --config 라벨과 글자단위 일치해야 함(불일치 시 $/Mtok=NaN).
# ⚠️ 아래는 placeholder — 최종 인스턴스/노드수 확정 후 조정.
#    inter-node 1P1D면 2개 인스턴스 합으로, 1P3D면 점유 인스턴스 수만큼 합산.
COST_PER_HR = {
    "T1": _INSTANCE_HR["g6.xlarge"] * 2,   # inter 1P1D 베이스라인 (prefill + decode 노드)
    "T2": _INSTANCE_HR["g6.12xlarge"],     # 1P3D — 실제 점유로 조정
    "T3": _INSTANCE_HR["g6.12xlarge"],     # 비대칭 TP/PP
    "T4": _INSTANCE_HR["g6.12xlarge"],     # intra 반복
}


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


def dollar_per_m_tokens(stats: dict, config: str) -> float:
    """$/M output tokens using official output_throughput and on-demand price."""
    thr = stats.get("out_tok_s")
    if not thr or thr != thr:
        return float("nan")
    cost_hr = COST_PER_HR.get(config, float("nan"))
    m_tok_hr = thr * 3600 / 1e6   # tokens/s → M tokens/hr
    return cost_hr / m_tok_hr if m_tok_hr else float("nan")


def print_table(all_stats: dict[str, dict[str, dict]]) -> None:
    header = (
        f"{'config':<8} {'point':<28} {'n_ok':>6} {'fail%':>6}"
        f" {'ttft_p50':>9} {'ttft_p99':>9} {'tpot_p50':>9} {'tpot_p99':>9}"
        f" {'itl_p99':>8} {'e2el_p99':>9} {'out_tok/s':>10} {'$/Mtok':>8}"
        f" {'kv_p50':>8} {'kv_p99':>8} {'pf_rps':>7} {'dc_rps':>7} {'pf_tps':>8} {'dc_tps':>8}"
    )
    print(header)
    print("-" * len(header))

    for config in sorted(all_stats):
        for point_id in sorted(all_stats[config]):
            s = all_stats[config][point_id]
            if s.get("n_ok", 0) == 0:
                print(f"{config:<8} {point_id:<28} {'NO DATA':>6}")
                continue
            dpm = dollar_per_m_tokens(s, config)
            fail_pct = s.get("fail_rate")
            fail_s = f"{fail_pct*100:>5.1f}%" if isinstance(fail_pct, (int, float)) and fail_pct == fail_pct else f"{'n/a':>6}"
            print(
                f"{config:<8} {point_id:<28} {s['n_ok']:>6} {fail_s}"
                f" {_fmt(s.get('ttft_p50_ms'),9)} {_fmt(s.get('ttft_p99_ms'),9)}"
                f" {_fmt(s.get('tpot_p50_ms'),9)} {_fmt(s.get('tpot_p99_ms'),9)}"
                f" {_fmt(s.get('itl_p99_ms'),8)} {_fmt(s.get('e2el_p99_ms'),9)}"
                f" {_fmt(s.get('out_tok_s'),10)} {_fmt(dpm,8,3)}"
                f" {_fmt(s.get('kv_transfer_p50_ms'),8,2)} {_fmt(s.get('kv_transfer_p99_ms'),8,2)}"
                f" {_fmt(s.get('prefill_rps'),7,2)} {_fmt(s.get('decode_rps'),7,2)}"
                f" {_fmt(s.get('prefill_tps'),8,0)} {_fmt(s.get('decode_tps'),8,0)}"
            )


def plot_comparison(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """One plot per (prefill_len, decode_len) pair: TTFT/TPOT/$ vs rate for all configs."""
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"prefill={pl} decode={dl}")

        for config, rate_stats in sorted(config_data.items()):
            rate_stats.sort(key=lambda x: x[0])
            rates  = [x[0] for x in rate_stats]
            ttft50 = [_num(x[1].get("ttft_p50_ms")) for x in rate_stats]
            ttft99 = [_num(x[1].get("ttft_p99_ms")) for x in rate_stats]
            tpot50 = [_num(x[1].get("tpot_p50_ms")) for x in rate_stats]
            dpm    = [_num(dollar_per_m_tokens(x[1], config)) for x in rate_stats]

            axes[0].plot(rates, ttft50, marker="o", label=f"{config} p50")
            axes[0].plot(rates, ttft99, marker="x", linestyle="--", label=f"{config} p99")
            axes[1].plot(rates, tpot50, marker="o", label=config)
            axes[2].plot(rates, dpm, marker="o", label=config)

        axes[0].set_title("TTFT (ms)");        axes[0].set_xlabel("rate (req/s)"); axes[0].legend(fontsize=7)
        axes[1].set_title("TPOT p50 (ms/tok)"); axes[1].set_xlabel("rate (req/s)"); axes[1].legend(fontsize=7)
        axes[2].set_title("$/M tokens (OD)");   axes[2].set_xlabel("rate (req/s)"); axes[2].legend(fontsize=7)

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
            all_stats[config][point_id] = s

    if not all_stats:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    print_table(all_stats)

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
