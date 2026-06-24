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


# 결과 파일명 — 자기설명적(폴더만 봐도 내용 알게). point_id = p{prefill}_d{decode}_r{rate}.
#   sweep.py와 반드시 동일하게 유지(쓰기=sweep, 읽기=여기).
F_BENCH = "latency_throughput"        # 공식 benchmark_serving: TTFT/TPOT/ITL/E2EL/throughput
F_PROM = "perside_rps"                # orchestrator ctx/gen 완료 카운터 → prefill/decode RPS
F_PERF = "kv_transfer"                # /perf_metrics: per-request KV전송 시간
F_BATCH = "perside_batch_kv_backlog"  # 워커 /metrics 집계: 배치수·KV풀·backlog
F_LIVE = "timeseries"                 # 1Hz per-tick 스냅샷(.jsonl) — 시간순 동역학


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


def _raw_dir(config_dir: Path) -> Path:
    """원본 per-point JSON 위치. 새 레이아웃은 <config>/raw/, 옛 평면 레이아웃도 그대로 지원."""
    r = config_dir / "raw"
    return r if r.exists() else config_dir


def list_points(config_dir: Path) -> list[str]:
    """config의 latency_throughput_<point>.json들에서 point_id 목록 추출 (raw/ 우선)."""
    rd = _raw_dir(config_dir)
    if not rd.exists():
        return []
    return sorted(p.stem[len(F_BENCH) + 1:] for p in rd.glob(f"{F_BENCH}_*.json"))


def load_bench(config_dir: Path, point_id: str) -> dict:
    """bench_{point_id}.json (공식 benchmark_serving --save-result)에서 집계 메트릭을 읽음.
    percentile 키는 sweep가 넘긴 --metric-percentiles(50,99) 기준(p50_*_ms / p99_*_ms).
    파일 없음/깨짐/실패면 {'n_ok':0} → 표에 NO DATA."""
    bf = _raw_dir(config_dir) / f"{F_BENCH}_{point_id}.json"
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
        "duration_s":  d.get("duration"),      # main run 실제 부하시간(초) — per-side rps 분모(희석 방지)
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


def _timing(side_obj) -> dict:
    """ctx_/gen_perf_metrics → timing_metrics (perf_metrics 래퍼 유무 둘 다 대응)."""
    pm = (side_obj or {}).get("perf_metrics") or side_obj or {}
    return pm.get("timing_metrics") or {}


def _agg3(vals: list) -> dict:
    """list → {mean, p50, p99} (값 없으면 빈 dict)."""
    xs = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not xs:
        return {}
    return {"mean": sum(xs) / len(xs), "p50": _p(xs, 50), "p99": _p(xs, 99)}


# 한 요청 생애 단계 (perf_metrics 같은 시계). (코드검증: workflow disagg-metrics-audit, 인용 kv_transfer_*.json)
#   prefill_compute = ctx 첫토큰 − ctx first_scheduled      (순수 prefill 연산)
#   prefill_queue   = ctx first_scheduled − ctx arrival      (prefill 큐 대기)
#   kv_transfer     = gen transfer_end − transfer_start      (inter-node KV전송, = "버퍼 대기")
#   handoff         = gen arrival − ctx 첫토큰               (orchestrator relay)
#   decode_queue    = gen first_scheduled − gen arrival      (decode 큐 대기 = 보통 지배적!)
#   decode          = gen 마지막토큰 − gen 첫토큰            (순수 decode, 첫토큰 이후)
#   ttft_recon      = disagg 첫토큰 − disagg 도착            (재구성 TTFT; 클라 ttft와 교차검증)
#   e2e             = gen 마지막토큰 − disagg 도착           (요청 전체)
# ⚠️ 단계 합 ≈ e2e (정확한 분할 아님 — 일부 overlap/생략 segment 있음, glossary 참고).
_PERF_STAGES = ("prefill_compute", "prefill_queue", "kv_transfer", "handoff",
                "decode_queue", "decode", "ttft_recon", "e2e")


def load_perf_breakdown(config_dir: Path, point_id: str) -> dict:
    """kv_transfer_{point}.json(/perf_metrics)에서 요청별 지연을 단계분해(초, 같은 시계) → 각 단계 mean/p50/p99.
       warmup(8토큰, decode≤1s)은 걸러 측정 요청만. 파일 없으면 빈 dict.
       (단계 정의·코드검증은 위 _PERF_STAGES 주석 / workflow audit 참조.)"""
    pf = _raw_dir(config_dir) / f"{F_PERF}_{point_id}.json"
    if not pf.exists():
        return {}
    try:
        data = json.loads(pf.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    cols: dict = {k: [] for k in _PERF_STAGES}
    reused = missed = 0
    WARMUP_DECODE_MAX_S = 1.0   # warmup(decode 8토큰)≈0.3s → 측정(decode≥수초)만 채택
    for e in data:
        if not isinstance(e, dict):
            continue
        ctx, gen = _timing(e.get("ctx_perf_metrics")), _timing(e.get("gen_perf_metrics"))
        gft, glt = gen.get("first_token_time"), gen.get("last_token_time")
        if not (isinstance(gft, (int, float)) and isinstance(glt, (int, float))):
            continue
        if (glt - gft) <= WARMUP_DECODE_MAX_S:   # warmup 제외
            continue
        cols["decode"].append(glt - gft)
        ca, cfs, cft = ctx.get("arrival_time"), ctx.get("first_scheduled_time"), ctx.get("first_token_time")
        ga, gfs = gen.get("arrival_time"), gen.get("first_scheduled_time")
        ts, te = gen.get("kv_cache_transfer_start"), gen.get("kv_cache_transfer_end")
        da, dft = e.get("disagg_server_arrival_time"), e.get("disagg_server_first_token_time")
        if isinstance(cft, (int, float)) and isinstance(cfs, (int, float)):
            cols["prefill_compute"].append(cft - cfs)
        if isinstance(cfs, (int, float)) and isinstance(ca, (int, float)):
            cols["prefill_queue"].append(cfs - ca)
        if isinstance(te, (int, float)) and isinstance(ts, (int, float)):
            cols["kv_transfer"].append(te - ts)
        if isinstance(ga, (int, float)) and isinstance(cft, (int, float)):
            cols["handoff"].append(ga - cft)
        if isinstance(gfs, (int, float)) and isinstance(ga, (int, float)):
            cols["decode_queue"].append(gfs - ga)
        if isinstance(dft, (int, float)) and isinstance(da, (int, float)):
            cols["ttft_recon"].append(dft - da)
        if isinstance(da, (int, float)):
            cols["e2e"].append(glt - da)
        km = ((e.get("gen_perf_metrics") or {}).get("perf_metrics") or {}).get("kv_cache_metrics") or {}
        reused += km.get("num_reused_blocks", 0)
        missed += km.get("num_missed_blocks", 0)
    out: dict = {}
    for stage in _PERF_STAGES:
        for stat, v in _agg3(cols[stage]).items():
            out[f"{stage}_{stat}_s"] = v
    # 기존 표/headline 호환 키 (단계 p50을 그대로 매핑)
    out["prefill_s"] = out.get("prefill_compute_p50_s")
    out["kv_transfer_s"] = out.get("kv_transfer_p50_s")
    out["decode_stage_s"] = out.get("decode_p50_s")
    out["e2e_stage_s"] = out.get("e2e_p50_s")
    if reused + missed:
        out["block_reuse_ratio"] = reused / (reused + missed)
    return {k: v for k, v in out.items() if v is not None}


def load_prom(config_dir: Path, point_id: str, mean_pt: float | None, mean_ct: float | None,
              bench_duration: float | None = None) -> dict:
    """prom_{point_id}.json (measured-윈도우 전/후 스냅샷)에서 per-side RPS를 계산하고,
    강제된 토큰길이(mean_pt/mean_ct)로 per-side TPS를 파생.
    - prefill_rps = (ctx_after − ctx_before)/T, decode_rps = (gen_after − gen_before)/T
      (ctx_/gen_completed_requests_total — prom_scrape.py 참조)
    - 분모 T: ⚠️ prom window는 benchmark_serving startup+초기테스트까지 포함해 실제 부하보다 길다(rps 희석,
      rate↑일수록 심함). 그래서 bench_duration(main run 실측 부하시간)이 있으면 그걸 분모로 → 희석 제거.
      (count엔 초기테스트 1건이 섞일 수 있으나 N 크면 무시 수준.) 없으면 window_s로 폴백.
    - per-side TPS는 공식 토큰 카운터가 없어(LEARNING_NOTES.md §병렬화·KV전송·per-side 6번) RPS × 토큰수로 파생.
    파일 없으면(=스크레이프 비활성/orchestrator 미노출) 빈 dict → 표에 'n/a'."""
    pf = _raw_dir(config_dir) / f"{F_PROM}_{point_id}.json"
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
    # 희석 방지: 실제 부하시간(bench main run)이 있으면 그걸 분모로.
    if isinstance(bench_duration, (int, float)) and bench_duration > 0:
        window_s = bench_duration
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
    # prefill이 decode보다 초당 몇 개 더 끝내나 = backlog가 쌓이는 속도(양수=decode가 못 따라감).
    if pf_rps is not None and dc_rps is not None:
        out["accum_rps"] = pf_rps - dc_rps
    return out


def load_batch(config_dir: Path, point_id: str) -> dict:
    """batch_<point>.json (워커 /metrics 샘플) → per-side 배치수·KV사용 (mean). 없으면 {}.
    pf_bsz=prefill 평균 배치(numContextRequests), dc_bsz=decode 평균 배치(numGenRequests),
    dc_kv_pct=decode KV풀 사용률(usedNumBlocks/maxNumBlocks)."""
    p = _raw_dir(config_dir) / f"{F_BATCH}_{point_id}.json"
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
    # peak(max)도 노출 — mean은 ramp/idle로 희석되니 천장값(=VRAM·KV풀 한계 신호)을 별도로.
    if d.get("decode_batch_max") is not None:
        out["dc_bsz_max"] = d["decode_batch_max"]
    if d.get("decode_kv_used_frac_mean") is not None:
        out["dc_kv_pct"] = d["decode_kv_used_frac_mean"] * 100
    if d.get("decode_kv_used_frac_max") is not None:
        out["dc_kv_pct_max"] = d["decode_kv_used_frac_max"] * 100
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


# 표/CSV 공통 컬럼 스펙: (표시이름, stats키, 최소너비, 소수자리).
# 이름은 풀네임 = "무엇 + 집계방식(median/mean/p99) + 단위"가 이름만 봐도 보이게.
# 너비는 이름보다 짧으면 format_tables가 자동으로 이름 길이에 맞춰 늘림.
_TABLE_GROUPS = [
    ("① 한 요청 시간 단계분해 (perf_metrics, 요청별 median, 초) — e2e ≈ prefill + kv전송 + decode", [
        ("prefill_time_median_s", "prefill_s", 0, 3),
        ("kv_transfer_time_median_s", "kv_transfer_s", 0, 3),
        ("decode_time_median_s", "decode_stage_s", 0, 3),
        ("e2e_time_median_s", "e2e_stage_s", 0, 3),
    ]),
    ("①b 단계분해 상세 (perf_metrics, 초, p50) — TTFT는 보통 decode_queue(=decode 슬롯 대기)가 지배. 단계합≈e2e(정확분할 아님)", [
        ("prefill_compute_p50_s", "prefill_compute_p50_s", 0, 3),
        ("prefill_queue_p50_s", "prefill_queue_p50_s", 0, 3),
        ("kv_transfer_p50_s", "kv_transfer_p50_s", 0, 3),
        ("handoff_p50_s", "handoff_p50_s", 0, 3),
        ("decode_queue_p50_s", "decode_queue_p50_s", 0, 3),
        ("decode_queue_mean_s", "decode_queue_mean_s", 0, 3),
        ("decode_queue_p99_s", "decode_queue_p99_s", 0, 3),
        ("decode_p50_s", "decode_p50_s", 0, 3),
        ("ttft_recon_p50_s", "ttft_recon_p50_s", 0, 3),
    ]),
    ("② 공식 지연 (benchmark_serving, 클라이언트 측정, 초) — TTFT=첫토큰까지·TPOT=토큰당·E2E=요청끝까지", [
        ("requests_ok", "n_ok", 0, 0), ("fail_pct", "fail_pct", 0, 1),
        ("ttft_median_s", "ttft_p50_s", 0, 3), ("ttft_p99_s", "ttft_p99_s", 0, 3),
        ("tpot_median_s", "tpot_p50_s", 0, 4), ("e2e_client_p99_s", "e2el_p99_s", 0, 2),
    ]),
    ("③ 처리량", [
        ("output_tokens_per_sec", "out_tok_s", 0, 1),
    ]),
    ("④ prefill vs decode 완료 속도·차이 (측정창 평균, [계산]) — 차이>0이면 decode가 못 따라가 backlog 쌓임", [
        ("prefill_completed_reqs_per_sec", "prefill_rps", 0, 3),
        ("decode_completed_reqs_per_sec", "decode_rps", 0, 3),
        ("prefill_minus_decode_per_sec", "accum_rps", 0, 3),
        ("prefill_tokens_per_sec_approx", "prefill_tps", 0, 0),
        ("decode_tokens_per_sec_approx", "decode_tps", 0, 0),
    ]),
    ("⑤ prefill vs decode 동시처리·KV·대기 (1Hz 샘플; mean=평균, max=peak 천장) — mean은 ramp/idle로 희석되니 max도 봄", [
        ("prefill_concurrent_reqs_mean", "pf_bsz", 0, 2),
        ("decode_concurrent_reqs_mean", "dc_bsz", 0, 2),
        ("decode_concurrent_reqs_max", "dc_bsz_max", 0, 0),
        ("decode_kvpool_used_pct_mean", "dc_kv_pct", 0, 1),
        ("decode_kvpool_used_pct_max", "dc_kv_pct_max", 0, 1),
        ("backlog_waiting_reqs_mean", "backlog", 0, 1),
        ("backlog_waiting_reqs_max", "backlog_max", 0, 0),
    ]),
]


def metric_glossary() -> str:
    """각 지표 = 어디서 왔나. [공식]=benchmark_serving, [서버]=워커/orchestrator raw, [계산]=우리 코드 식."""
    return "\n".join([
        "── 지표 = 무엇 / 출처 / 식  ([공식]=benchmark_serving · [서버]=perf_metrics·워커/metrics · [계산]=우리 코드). 시간=초 ──",
        "[① 한 요청 시간 단계분해 — perf_metrics, 요청별 median]",
        "prefill_time_median_s         [서버] ctx 첫토큰 − ctx 도착 (prefill 단계)",
        "kv_transfer_time_median_s     [서버] kv_cache_transfer_end − start (inter-node KV전송 단계)",
        "decode_time_median_s          [서버] gen 마지막토큰 − gen 첫토큰 (decode 단계, 첫토큰 이후)",
        "e2e_time_median_s             [서버] gen 마지막토큰 − disagg 도착 (요청 전체) ≈ 위 셋의 합",
        "   ※ warmup(8토큰, decode≤1s) 제외하고 측정 요청만 집계",
        "[①b 단계 상세 — 한 요청 TTFT가 어디서 대기/소비되나 (perf_metrics 같은 시계)]",
        "prefill_compute_p50_s   [서버] ctx first_token − first_scheduled (순수 prefill 연산, 작음·일정)",
        "prefill_queue_p50_s     [서버] ctx first_scheduled − arrival (prefill 큐 대기; 고rate서 폭증=backpressure)",
        "kv_transfer_p50_s       [서버] = 버퍼 대기 (inter-node KV전송, 작음)",
        "handoff_p50_s           [서버] gen arrival − ctx first_token (orchestrator relay, 작음)",
        "decode_queue_p50/mean/p99_s [서버] gen first_scheduled − arrival (decode 슬롯 대기; 보통 TTFT 지배)",
        "ttft_recon_p50_s        [서버] disagg first_token − arrival (재구성 TTFT; 클라 ttft와 일치로 시계검증)",
        "   ※ 단계 합 ≈ e2e (overlap 있어 정확 분할 아님). TPOT 재구성 = decode_p50_s / (출력토큰−1).",
        "[② 공식 지연 — benchmark_serving, 클라이언트가 측정한 SLO]",
        "requests_ok / fail_pct        [공식] 완료 요청수 / 실패율(%)",
        "ttft_median_s / ttft_p99_s    [공식] 첫 토큰까지 (median / 99퍼센타일)",
        "tpot_median_s                 [공식] 토큰당 시간(첫토큰 제외, median). ※E2E ≈ ttft + tpot×출력토큰수",
        "e2e_client_p99_s              [공식] 요청 끝까지 (99퍼센타일)",
        "output_tokens_per_sec         [공식] 생성토큰 합 / 측정시간",
        "[③ prefill vs decode 완료 속도·차이 — orchestrator 완료 카운터 / 실제 부하시간(bench duration)]",
        "  ※분모=benchmark_serving 'duration'(main run 실측). prom window는 startup·초기테스트 포함해 길어 rps 희석 → 이걸로 보정.",
        "prefill_completed_reqs_per_sec [계산] (측정후−측정전 ctx_completed)/부하시간 = prefill이 초당 끝낸 요청수",
        "decode_completed_reqs_per_sec  [계산] (측정후−측정전 gen_completed)/부하시간 = decode가 초당 끝낸 요청수",
        "prefill_minus_decode_per_sec   [계산] 위 둘의 차이 = backlog 쌓이는 속도 (양수=decode가 못 따라감)",
        "prefill_tokens_per_sec_approx  [계산·근사] prefill_completed × 입력길이(ISL). ※토큰 카운터 없어 곱셈 근사",
        "decode_tokens_per_sec_approx   [계산·근사] decode_completed × 출력길이(OSL)",
        "[④ 동시처리·KV·대기 — 워커 /metrics, 1Hz 샘플. mean=평균(ramp/idle 희석), max=peak(천장)]",
        "prefill_concurrent_reqs_mean   [서버] prefill 워커 동시 처리 요청수(numContextRequests) 평균",
        "decode_concurrent_reqs_mean/max[서버] decode 워커 동시 처리 요청수(numGenRequests) 평균/최대(=VRAM 천장)",
        "decode_kvpool_used_pct_mean/max[서버] decode KV풀 사용률(usedNumBlocks/maxNumBlocks×100) 평균/최대",
        "backlog_waiting_reqs_mean/max  [계산] (ctx_completed−gen_completed) 평균/최대 = decode 대기 중인 요청수(쌓인 양)",
    ])


def format_tables(all_stats: dict[str, dict[str, dict]]) -> str:
    """4개 서브-표를 문자열로 렌더(터미널 출력 + 파일 저장 공용). 행 키 = (config, point)."""
    keys = [(c, p) for c in sorted(all_stats) for p in sorted(all_stats[c])]
    lines: list[str] = []
    for title, cols in _TABLE_GROUPS:
        # 컬럼 너비 = max(스펙 너비, 풀네임 길이) → 긴 이름도 안 깨지고 정렬됨.
        widths = [max(w, len(n)) for n, _, w, _ in cols]
        hdr = f"{'config':<8} {'point':<24}" + "".join(f" {n:>{cw}}" for (n, _, _, _), cw in zip(cols, widths))
        lines += [f"\n── {title} ──", hdr, "-" * len(hdr)]
        for c, p in keys:
            s = all_stats[c][p]
            row = f"{c:<8} {p:<24}"
            for (_, key, _, prec), cw in zip(cols, widths):
                row += f" {_fmt(s.get(key), cw, prec)}"
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


# REPORT.md 핵심표 = 꼭 보는 지표만(나머지 전부는 data.csv). (표시명, stats키, 소수자리)
# 단계분해(초) 중심 — 한 요청 시간이 prefill/KV전송/decode 어디에 쓰이나 + 처리량·decode배치·backlog.
_HEADLINE = [
    ("prefill_time_median_s", "prefill_s", 3),
    ("kv_transfer_time_median_s", "kv_transfer_s", 3),
    ("decode_time_median_s", "decode_stage_s", 3),
    ("e2e_time_median_s", "e2e_stage_s", 3),
    ("output_tokens_per_sec", "out_tok_s", 1),
    ("decode_concurrent_reqs_mean", "dc_bsz", 1),
    ("backlog_waiting_reqs_mean", "backlog", 1),
    ("prefill_minus_decode_per_sec", "accum_rps", 3),
]


def _headline_md(stats_one: dict) -> str:
    """핵심 지표만 markdown 표 1개로(에디터/GitHub서 표로 렌더)."""
    head = "| point | " + " | ".join(n for n, _, _ in _HEADLINE) + " |"
    sep = "|" + "---|" * (len(_HEADLINE) + 1)
    rows = [head, sep]
    for config in sorted(stats_one):
        for p in sorted(stats_one[config]):
            s = stats_one[config][p]
            cells = []
            for _, key, prec in _HEADLINE:
                v = s.get(key)
                cells.append(f"{v:.{prec}f}" if isinstance(v, (int, float)) and v == v else "n/a")
            rows.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _file_index() -> str:
    return "\n".join([
        "REPORT.md      이 파일 — 핵심표 + 지표뜻 + 폴더안내",
        "data.csv       전체 수치(모든 메트릭, RAW, 논문용)",
        "plots/         timeseries_<pt>.png(포인트별 시간순) · grid_compare_*.png(rate별 비교)",
        "raw/           원본 JSON 전부(기록용): latency_throughput / perside_rps / kv_transfer /",
        "               perside_batch_kv_backlog / timeseries.jsonl",
        "metadata.json  설정(모델·ctx/gen TP·PP·placement)",
        ".done_* .failed_*   재실행 스킵 마커",
    ])


def write_report(config: str, config_dir: Path, stats_one: dict) -> None:
    """config 폴더에 자체완결 REPORT.md(핵심표+지표뜻+안내) + data.csv(전체) 작성."""
    meta = {}
    mp = config_dir / "metadata.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            pass
    c, g = meta.get("context", {}), meta.get("generation", {})
    topo = (f"{c.get('num_instances','?')}P(tp{c.get('tp','?')},pp{c.get('pp','?')})"
            f" + {g.get('num_instances','?')}D(tp{g.get('tp','?')},pp{g.get('pp','?')})")
    md = [
        f"# {config} — TRT-LLM PD 분리 결과", "",
        f"- 모델: {meta.get('model','?')}",
        f"- 토폴로지: {topo}  ·  배치: {meta.get('placement','?')}-node  ·  KV전송: {meta.get('cache_transceiver_backend','?')}",
        "- 부하: 공식 benchmark_serving (open-loop, Poisson). point = p{prefill}_d{decode}_r{rate}", "",
        "## 핵심 결과", _headline_md(stats_one), "",
        "> 핵심표는 단계분해(초)·처리량·decode배치·backlog. 공식 TTFT/TPOT/E2EL·per-side rps·전체 수치 = `data.csv` · 시간순 = `plots/timeseries_*.png` · rate별 = `plots/grid_compare_*.png`", "",
        "## 지표 뜻 / 출처·식", "```", metric_glossary(), "```", "",
        "## 이 폴더 안내", "```", _file_index(), "```",
    ]
    (config_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    write_csv(stats_one, config_dir / "data.csv")


def plot_timeseries(config: str, point_id: str, config_dir: Path) -> None:
    """[grid 포인트 1개의 시간순 동역학] timeseries_<point>.jsonl(1Hz per-tick)를 읽어
    3패널 시계열을 그 config 폴더에 저장:
      ① 동시 배치수(prefill/decode)가 시간에 따라 어떻게 변하나
      ② backlog = prefill 끝났는데 decode 대기 중인 요청수(프리필 큐에 쌓인 양)
      ③ 이 실험 완료수(0부터, 누적 아님) — 총 처리가 어떻게 진행되나
    (배치수는 enable_iter_perf_stats:true여야 채워짐. null이면 그 패널만 'n/a' 표시.)"""
    if not HAS_MPLOT:
        return
    fp = _raw_dir(config_dir) / f"{F_LIVE}_{point_id}.jsonl"
    if not fp.exists():
        return
    rows = []
    for line in fp.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return

    def xy(*keys):  # 값이 None 아닌 첫 키의 (t, value). 신/구 키 모두 대응(구 trace 호환)
        xs, ys = [], []
        for r in rows:
            v = next((r[k] for k in keys if r.get(k) is not None), None)
            if v is not None:
                xs.append(r.get("t"))
                ys.append(v)
        return xs, ys

    # 라벨은 영어로 (컨테이너 matplotlib에 한글 폰트 없어 □ 깨짐 + 논문용 적합).
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{config}  {point_id}  (time-series, t = s since measure start)")

    # ① concurrent batch
    ax = axes[0]
    pf_x, pf_y = xy("prefill_batch", "pf_bsz")
    dc_x, dc_y = xy("decode_batch", "dc_bsz")
    if pf_x:
        ax.plot(pf_x, pf_y, marker=".", label="prefill batch")
    if dc_x:
        ax.plot(dc_x, dc_y, marker=".", label="decode batch")
    if not pf_x and not dc_x:
        ax.text(0.5, 0.5, "batch n/a\n(set enable_iter_perf_stats, re-measure)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        ax.legend(fontsize=8)
    ax.set_title("concurrent batch (req)"); ax.set_xlabel("t (s)"); ax.set_ylabel("batch")

    # ② queues over time: prefill-side queue (대기) + backlog (prefill done, decode pending)
    ax = axes[1]
    qx, qy = xy("prefill_queue")                 # ctx_total - ctx_completed (재수집한 run만 존재)
    bx, by = xy("backlog")
    if qx:
        ax.plot(qx, qy, marker=".", color="tab:red", label="prefill_queue (arrived, not prefilled)")
    if bx:
        ax.plot(bx, by, marker=".", color="tab:green", label="backlog (prefilled, not decoded)")
    if qx or bx:
        ax.legend(fontsize=8)
    if not qx:
        ax.text(0.5, 0.02, "prefill_queue: re-run for this (not in old trace)",
                ha="center", va="bottom", transform=ax.transAxes, fontsize=7, color="gray")
    ax.set_title("queues over time (waiting requests)"); ax.set_xlabel("t (s)"); ax.set_ylabel("req")

    # ③ cumulative completions
    ax = axes[2]
    cx, cy = xy("prefill_done", "ctx_done")
    gx, gy = xy("decode_done", "gen_done")
    if cx:
        ax.plot(cx, cy, marker=".", label="prefill_done (this run)")
    if gx:
        ax.plot(gx, gy, marker=".", linestyle="--", label="decode_done (this run)")
    if cx or gx:
        ax.legend(fontsize=8)
    ax.set_title("completions this run (slope = rps)"); ax.set_xlabel("t (s)"); ax.set_ylabel("requests")

    fig.tight_layout()
    plots = config_dir / "plots"
    plots.mkdir(exist_ok=True)
    fname = plots / f"{F_LIVE}_{point_id}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_comparison(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    """[grid 비교] (prefill_len, decode_len) 쌍마다 1장, 4패널 vs rate(여러 config 겹쳐):
    TTFT / per-side 완료 rps(prefill vs decode) / output throughput / batch & backlog.
    = rate(grid)에 따라 메트릭이 어떻게 변하는지. (포인트별 시간순은 plot_timeseries 참조.)"""
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
            ttft50 = [_num(x[1].get("ttft_p50_s")) for x in rate_stats]
            ttft99 = [_num(x[1].get("ttft_p99_s")) for x in rate_stats]
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

        axes[0].set_title("TTFT (s)");         axes[0].set_xlabel("rate (req/s)"); axes[0].legend(fontsize=7)
        axes[1].set_title("per-side completion rps (prefill vs decode)"); axes[1].set_xlabel("rate (req/s)"); axes[1].legend(fontsize=7)
        axes[2].set_title("output throughput (tok/s)"); axes[2].set_xlabel("rate (req/s)"); axes[2].legend(fontsize=7)
        axes[3].set_title("batch & backlog (req)"); axes[3].set_xlabel("rate (req/s)"); axes[3].legend(fontsize=7)

        fig.tight_layout()
        fname = out_dir / f"grid_compare_p{pl}_d{dl}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  saved {fname}")


def plot_latency_decomp(all_stats: dict, out_dir: Path) -> None:
    """[compare #1] 모든 포인트의 지연 단계분해를 mean/p50/p99 묶음막대로.
    한 요청 시간이 prefill_compute / prefill_queue / kv_transfer(버퍼대기) / decode_queue(슬롯대기, 보통 지배)
    / decode / TTFT / TPOT / e2e 어디에 쓰이나. (perf_metrics, warmup 제외.)"""
    if not HAS_MPLOT:
        return
    pts = [(c, p) for c in sorted(all_stats) for p in sorted(all_stats[c])]
    if not pts:
        return
    labels = [p for _, p in pts]

    def series(prefix):
        return [{k: all_stats[c][p].get(f"{prefix}_{k}_s") for k in ("mean", "p50", "p99")} for c, p in pts]

    def tpot_series():  # decode/(OSL-1)로 파생
        out = []
        for c, p in pts:
            s = all_stats[c][p]
            try:
                _, dl, _ = parse_point_id(p)
            except Exception:
                dl = None
            out.append({k: (s.get(f"decode_{k}_s") / (dl - 1)) if (dl and dl > 1 and isinstance(s.get(f"decode_{k}_s"), (int, float))) else None
                        for k in ("mean", "p50", "p99")})
        return out

    panels = [
        ("prefill_compute (s)", series("prefill_compute")),
        ("prefill_queue wait (s)", series("prefill_queue")),
        ("kv_transfer = buffer wait (s)", series("kv_transfer")),
        ("decode_queue wait (s) [usually dominant]", series("decode_queue")),
        ("decode (s)", series("decode")),
        ("TTFT (s)", series("ttft_recon")),
        ("TPOT (s)", tpot_series()),
        ("e2e (s)", series("e2e")),
    ]
    x = np.arange(len(labels))
    w = 0.27
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle("latency decomposition per point — mean / p50 / p99  (perf_metrics, warmup-filtered; stages ~sum to e2e, not exact)")
    for ax, (title, data) in zip(axes.flat, panels):
        m = [_num(d.get("mean")) for d in data]
        p5 = [_num(d.get("p50")) for d in data]
        p9 = [_num(d.get("p99")) for d in data]
        ax.bar(x - w, m, w, label="mean")
        ax.bar(x, p5, w, label="p50")
        ax.bar(x + w, p9, w, label="p99")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=6); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "compare_latency_decomp.png"
    fig.savefig(fname, dpi=120); plt.close(fig)
    print(f"  saved {fname}")


def plot_perside_compare(all_stats: dict, out_dir: Path) -> None:
    """[compare #2] 모든 포인트의 per-side 처리량·동시성: prefill/decode rps, batch(mean+max), backlog(mean+max).
    1P1D에선 prefill_rps==decode_rps(=orchestrator 집계=단일 워커). batch는 mean_active 막대 + max 수염."""
    if not HAS_MPLOT:
        return
    pts = [(c, p) for c in sorted(all_stats) for p in sorted(all_stats[c])]
    if not pts:
        return
    labels = [p for _, p in pts]
    x = np.arange(len(labels))
    g = lambda c, p, k: _num(all_stats[c][p].get(k))

    def whisker(ax, xs, lo, hi):
        for xi, (a, b) in zip(xs, zip(lo, hi)):
            if a == a and b == b:
                ax.plot([xi, xi], [a, b], color="k", linewidth=0.8)
                ax.plot(xi, b, marker="_", color="k")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("per-side throughput & concurrency per point")
    w = 0.38
    # A: per-side completed rps
    ax = axes[0]
    pf = [g(c, p, "prefill_rps") for c, p in pts]
    dc = [g(c, p, "decode_rps") for c, p in pts]
    ax.bar(x - w / 2, pf, w, label="prefill_rps")
    ax.bar(x + w / 2, dc, w, label="decode_rps")
    ax.set_title("completed req/s per side  (1P1D: prefill==decode)")
    # B: concurrent batch (mean bar + max whisker)
    ax = axes[1]
    pfb = [g(c, p, "pf_bsz") for c, p in pts]
    dcb = [g(c, p, "dc_bsz") for c, p in pts]
    dcbmax = [g(c, p, "dc_bsz_max") for c, p in pts]
    ax.bar(x - w / 2, pfb, w, label="prefill_batch (mean)")
    ax.bar(x + w / 2, dcb, w, label="decode_batch (mean)")
    whisker(ax, x + w / 2, dcb, dcbmax)
    ax.set_title("concurrent batch  (bar=mean_active, whisker=decode max)")
    # C: backlog (mean bar + max whisker)
    ax = axes[2]
    bl = [g(c, p, "backlog") for c, p in pts]
    blmax = [g(c, p, "backlog_max") for c, p in pts]
    ax.bar(x, bl, w, label="backlog (mean)", color="tab:green")
    whisker(ax, x, bl, blmax)
    ax.set_title("backlog = prefill done, decode pending  (bar=mean, whisker=max)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "compare_perside.png"
    fig.savefig(fname, dpi=120); plt.close(fig)
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
            s.update(load_perf_breakdown(config_dir, point_id))             # 지연 단계분해(초): prefill/KV전송/decode/e2e
            s.update(load_prom(config_dir, point_id, mean_pt, mean_ct,
                               bench_duration=b.get("duration_s")))         # per-side RPS/TPS (분모=실제 부하시간)
            s.update(load_batch(config_dir, point_id))                      # per-side 배치수·KV사용 (있으면)
            # 공식(benchmark_serving) 지연은 ms → 사람이 보는 표는 초(_s)로. (raw json은 ms 그대로)
            for k in list(s):
                if k.endswith("_ms") and isinstance(s[k], (int, float)):
                    s[k[:-3] + "_s"] = s[k] / 1000.0
            all_stats[config][point_id] = s

    if not all_stats:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    # 터미널엔 전체 표 + glossary (즉시 확인용). 파일은 config 폴더별로 깔끔히.
    print(format_tables(all_stats))
    print("\n" + metric_glossary())

    if args.plot:
        # ① 포인트별 시간순 → 각 config의 plots/ (results/<config>/plots/timeseries_<pt>.png)
        for config in all_stats:
            for point_id in all_stats[config]:
                plot_timeseries(config, point_id, log_dir / config)
        # ② grid 비교(vs rate) → config 1개면 그 config의 plots/, 여러 개면 results/plots/(교차)
        comp_dir = (log_dir / next(iter(all_stats)) / "plots") if len(all_stats) == 1 else (log_dir / "plots")
        comp_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_stats, comp_dir)
        plot_latency_decomp(all_stats, comp_dir)   # compare #1: 지연 단계분해 mean/p50/p99
        plot_perside_compare(all_stats, comp_dir)  # compare #2: per-side rps·배치·backlog

    # 각 config 폴더에 자체완결 REPORT.md(핵심표+지표뜻+안내) + data.csv(전체) → 폴더만 열면 다 봄
    for config in all_stats:
        write_report(config, log_dir / config, {config: all_stats[config]})
        print(f"[analyze] {config} → results/{config}/REPORT.md  (+ data.csv · plots/ · raw/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=os.environ.get("EXP_LOG_DIR", "./results"))
    parser.add_argument("--configs", nargs="*", help="Which configs to analyze (default: all found)")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figures")
    args = parser.parse_args()
    main(args)
