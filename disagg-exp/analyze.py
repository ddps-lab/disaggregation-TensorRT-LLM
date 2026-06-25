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
    # plot 라벨은 전부 영어(논문 표준 워딩) → 기본 폰트 사용. (REPORT/표는 한글 markdown, 폰트 무관)
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
        "e2el_mean_ms": d.get("mean_e2el_ms"),   # 공식 client 요청 e2e 평균(paper용)
        "e2el_p50_ms": d.get("p50_e2el_ms"),
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
#   transfer_pool_wait = gen transfer_start − gen arrival   (순수 풀 대기, 전송 시작 전까지 — 전송시간과 안 겹침)
#   decode_queue    = gen first_scheduled − gen arrival      (총 gen-측 큐 = transfer_pool_wait + kv_transfer + 작은꼬리)
#   decode          = gen 마지막토큰 − gen 첫토큰            (순수 decode, 첫토큰 이후)
#   ttft_recon      = disagg 첫토큰 − disagg 도착            (재구성 TTFT = 위 도착~첫토큰 단계들의 합. 별개 atom 아님!)
#   e2e             = gen 마지막토큰 − disagg 도착           (요청 전체 = 모든 atom의 합)
# ⚠️ 겹침구조(전수검증): kv_transfer ⊂ decode_queue, ttft_recon ⊃ {prefill_queue, transfer_pool_wait, kv_transfer}.
#   비겹침 atom(합=e2e): prefill_queue + prefill_compute + handoff + transfer_pool_wait + kv_transfer + decode.
_PERF_STAGES = ("prefill_compute", "prefill_queue", "kv_transfer", "handoff",
                "transfer_pool_wait", "decode_queue", "decode", "ttft_recon", "e2e")


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
    ctx_done, gen_done = [], []   # 절대 완료시각: prefill=ctx 첫토큰, decode=gen 마지막토큰 → per-side 완료율
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
        gen_done.append(glt)                       # decode 완료시각(이 요청이 끝난 절대시각)
        ca, cfs, cft = ctx.get("arrival_time"), ctx.get("first_scheduled_time"), ctx.get("first_token_time")
        if isinstance(cft, (int, float)):
            ctx_done.append(cft)                   # prefill 완료시각(첫토큰=KV 준비 완료)
        ga, gfs = gen.get("arrival_time"), gen.get("first_scheduled_time")
        ts, te = gen.get("kv_cache_transfer_start"), gen.get("kv_cache_transfer_end")
        da, dft = e.get("disagg_server_arrival_time"), e.get("disagg_server_first_token_time")
        if isinstance(cft, (int, float)) and isinstance(cfs, (int, float)):
            cols["prefill_compute"].append(cft - cfs)
        if isinstance(cfs, (int, float)) and isinstance(ca, (int, float)):
            cols["prefill_queue"].append(cfs - ca)
        if isinstance(te, (int, float)) and isinstance(ts, (int, float)):
            cols["kv_transfer"].append(te - ts)
        if isinstance(ts, (int, float)) and isinstance(ga, (int, float)):
            cols["transfer_pool_wait"].append(ts - ga)   # 순수 풀 대기 = 전송 '시작' 전까지. transfer_pool_wait + kv_transfer + (작은꼬리) = decode_queue. → 전송시간과 안 겹침
        if isinstance(ga, (int, float)) and isinstance(cft, (int, float)):
            cols["handoff"].append(ga - cft)
        if isinstance(gfs, (int, float)) and isinstance(ga, (int, float)):
            cols["decode_queue"].append(gfs - ga)         # 총 gen-측 큐(=transfer_pool_wait + kv_transfer + 작은꼬리); 표/참고용
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
    # per-side 완료율(req/s) = 그 측이 끝낸 요청수 ÷ (첫완료~끝완료 구간). 1P1D라도 prefill이
    #   더 좁은 구간에 끝내므로 prefill_done_rps > decode_done_rps → decode가 병목임이 드러남.
    if len(ctx_done) >= 2:
        span = max(ctx_done) - min(ctx_done)
        if span > 0:
            out["prefill_done_rps"] = len(ctx_done) / span
    if len(gen_done) >= 2:
        span = max(gen_done) - min(gen_done)
        if span > 0:
            out["decode_done_rps"] = len(gen_done) / span
    if "prefill_done_rps" in out and "decode_done_rps" in out:
        out["prefill_minus_decode_done_rps"] = out["prefill_done_rps"] - out["decode_done_rps"]
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
    ("①b 비겹침 atom 분해 (perf_metrics, 초, p50) — 합=e2e(전수검증 오차 0.01%). transfer_pool_wait+kv_transfer=총 gen큐(decode_queue). TTFT/E2EL은 atom 합(별개 아님)", [
        ("prefill_queue_p50_s", "prefill_queue_p50_s", 0, 3),
        ("prefill_compute_p50_s", "prefill_compute_p50_s", 0, 3),
        ("handoff_p50_s", "handoff_p50_s", 0, 3),
        ("transfer_pool_wait_p50_s", "transfer_pool_wait_p50_s", 0, 3),
        ("kv_transfer_p50_s", "kv_transfer_p50_s", 0, 3),
        ("decode_p50_s", "decode_p50_s", 0, 3),
        ("decode_queue_total_p50_s", "decode_queue_p50_s", 0, 3),
        ("decode_queue_total_p99_s", "decode_queue_p99_s", 0, 3),
        ("ttft_recon_sum_p50_s", "ttft_recon_p50_s", 0, 3),
    ]),
    ("② 공식 지연 (benchmark_serving, 클라이언트 측정, 초) — TTFT=첫토큰까지·TPOT=토큰당·E2E=요청끝까지", [
        ("requests_ok", "n_ok", 0, 0), ("fail_pct", "fail_pct", 0, 1),
        ("ttft_median_s", "ttft_p50_s", 0, 3), ("ttft_p99_s", "ttft_p99_s", 0, 3),
        ("tpot_median_s", "tpot_p50_s", 0, 4), ("e2e_client_p99_s", "e2el_p99_s", 0, 2),
    ]),
    ("③ 처리량", [
        ("output_tokens_per_sec", "out_tok_s", 0, 1),
    ]),
    ("④ prefill vs decode 요청 완료율·차이 (perf_metrics 절대 완료시각, [계산]) — prefill>decode면 decode가 병목·backlog 쌓임", [
        ("prefill_completion_rate_per_sec", "prefill_done_rps", 0, 3),
        ("decode_completion_rate_per_sec", "decode_done_rps", 0, 3),
        ("prefill_minus_decode_completion_per_sec", "prefill_minus_decode_done_rps", 0, 3),
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
    ("⑥ 노드 사이징 — per-side 이용률 + 균형 P:D ([계산]) — prefill 컴퓨트 busy%=100×rps×prefill_compute, decode=KV풀 점유 max%. prefill 놀고 decode~100%면 D 늘려라", [
        ("prefill_compute_util_pct", "prefill_compute_util_pct", 0, 1),
        ("decode_kv_util_pct", "decode_kv_util_pct", 0, 1),
        ("prefill_capacity_reqs_per_sec", "prefill_capacity_reqs_per_sec", 0, 2),
        ("decode_capacity_reqs_per_sec", "decode_capacity_reqs_per_sec", 0, 2),
        ("balanced_decode_nodes_per_1_prefill", "balanced_decode_nodes_per_prefill", 0, 2),
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
        "[③ prefill vs decode 요청 완료율·차이 — perf_metrics 절대 완료시각 기반(per-request)]",
        "  ※prefill 완료=ctx 첫토큰(KV 준비됨), decode 완료=gen 마지막토큰. 각 측의 '첫완료~끝완료 구간'으로 나눔.",
        "  ※1P1D라도 prefill이 더 좁은 구간에 다 끝내므로 prefill>decode로 갈라짐 → decode가 율속(병목)임이 드러남.",
        "prefill_completion_rate_per_sec [계산] prefill 끝낸 요청수 ÷ (prefill 첫완료~끝완료 구간) = prefill이 초당 끝낸 요청수",
        "decode_completion_rate_per_sec  [계산] decode 끝낸 요청수 ÷ (decode 첫완료~끝완료 구간) = decode가 초당 끝낸 요청수",
        "prefill_minus_decode_completion_per_sec [계산] 위 둘의 차이 = prefill이 decode보다 빠른 정도(양수=decode 병목)",
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
    ("prefill_minus_decode_completion_per_sec", "prefill_minus_decode_done_rps", 3),
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
        "plots/         timeseries_<pt>.png(포인트별 시간순) · compare_latency_decomp.png(단계분해 mean/p50/p99) · compare_perside.png(per-side rps·배치·backlog)",
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
        "> 핵심표는 단계분해(초)·처리량·decode배치·backlog. 공식 TTFT/TPOT/E2EL·per-side rps·전체 수치 = `data.csv` · 시간순 = `plots/timeseries_*.png` · 포인트 비교 = `plots/compare_latency_decomp.png`(지연 단계분해)·`compare_perside.png`(per-side)", "",
        "## 지표 뜻 / 출처·식", "```", metric_glossary(), "```", "",
        "## 이 폴더 안내", "```", _file_index(), "```",
    ]
    (config_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    write_csv(stats_one, config_dir / "data.csv")


def plot_timeseries(config: str, point_id: str, config_dir: Path) -> None:
    """[포인트 1개의 시간순 동역학] timeseries_<point>.jsonl(per-tick)를 읽어 5패널 시계열 저장(라벨 영어):
      ① 현재 배치수(prefill/decode)  ② decode KV캐시 사용률(%)  ③ 요청큐 깊이(도착·prefill 전)
      ④ KV전송 풀 backlog(prefill끝·decode 대기)  ⑤ prefill·decode 누적 완료수(같은 축, 기울기=throughput)
    ②③는 신규 trace 키(decode_kv_frac / prefill_queue) 필요 — 구 trace엔 'needs re-run' 표시."""
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

    def xyk(num_key, den_key):  # 비율 시계열 → % (num/den, 둘 다 있을 때만)
        xs, ys = [], []
        for r in rows:
            n, d = r.get(num_key), r.get(den_key)
            if isinstance(n, (int, float)) and isinstance(d, (int, float)) and d:
                xs.append(r.get("t")); ys.append(100.0 * n / d)
        return xs, ys

    def note_rerun(ax):  # 이 trace에 없는 신규 지표 — 재런 필요 표시
        ax.text(0.5, 0.5, "needs re-run\n(not in this trace)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")

    # 라벨 전부 영어(논문 표준). 5패널: 배치 / KV사용률 / 요청큐깊이 / 전송풀 backlog / 누적완료
    fig, axes = plt.subplots(2, 3, figsize=(19, 8))
    fig.suptitle(f"{config}  {point_id}  (time-series, t = s since measure start)")
    axes = axes.flatten()

    # ① concurrent batch (현재 배치)
    ax = axes[0]
    pf_x, pf_y = xy("prefill_batch", "pf_bsz")
    dc_x, dc_y = xy("decode_batch", "dc_bsz")
    if pf_x:
        ax.plot(pf_x, pf_y, marker=".", label="prefill batch")
    if dc_x:
        ax.plot(dc_x, dc_y, marker=".", label="decode batch")
    if pf_x or dc_x:
        ax.legend(fontsize=8)
    else:
        note_rerun(ax)
    ax.set_title("Concurrent batch (requests)"); ax.set_xlabel("t (s)"); ax.set_ylabel("requests")

    # ② KV-cache utilization, decode (현재 KV캐시 사용률)
    ax = axes[1]
    kx, ky = xy("decode_kv_frac")
    if kx:
        ky = [v * 100.0 for v in ky]                       # frac 0~1 → %
    else:
        kx, ky = xyk("decode_kv_used", "decode_kv_max")    # used/max → %
    if kx:
        ax.plot(kx, ky, marker=".", color="tab:purple"); ax.set_ylim(0, 105)
    else:
        note_rerun(ax)
    ax.set_title("KV-cache utilization, decode (%)"); ax.set_xlabel("t (s)"); ax.set_ylabel("%")

    # ③ request-queue depth (요청큐 대기 수: 도착했으나 prefill 전)
    ax = axes[2]
    qx, qy = xy("prefill_queue")
    if qx:
        ax.plot(qx, qy, marker=".", color="tab:red")
    else:
        note_rerun(ax)
    ax.set_title("Request-queue depth (waiting to prefill)"); ax.set_xlabel("t (s)"); ax.set_ylabel("requests")

    # ④ transfer-pool backlog (KV전송 풀 대기 수: prefill끝·decode 대기)
    ax = axes[3]
    bx, by = xy("backlog")
    if bx:
        ax.plot(bx, by, marker=".", color="tab:green")
    else:
        note_rerun(ax)
    ax.set_title("KV transfer-pool backlog (prefilled, awaiting decode)"); ax.set_xlabel("t (s)"); ax.set_ylabel("requests")

    # ⑤ cumulative completions (prefill·decode 끝낸 누적 — 같은 그래프)
    ax = axes[4]
    cx, cy = xy("prefill_done", "ctx_done")
    gx, gy = xy("decode_done", "gen_done")
    if cx:
        ax.plot(cx, cy, marker=".", label="prefill done")
    if gx:
        ax.plot(gx, gy, marker=".", linestyle="--", label="decode done")
    if cx or gx:
        ax.legend(fontsize=8)
    else:
        note_rerun(ax)
    ax.set_title("Cumulative completions (slope = throughput)"); ax.set_xlabel("t (s)"); ax.set_ylabel("requests")

    axes[5].axis("off")   # 6번째 칸 비움 (5패널)

    fig.tight_layout()
    plots = config_dir / "plots"
    plots.mkdir(exist_ok=True)
    fname = plots / f"{F_LIVE}_{point_id}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_latency_decomp(all_stats: dict, out_dir: Path) -> None:
    """[compare #1] 요청 지연 분해 — 사용자 정의 5개 순차·절대 비겹침 단계 + E2EL(전체).
    [1] Queuing delay = prefill_queue (ca→cfs, 큐 대기)
    [2] TTFT = prefill_compute (cfs→cft, "큐에서 나와 prefill 시작→첫토큰", 큐 제외 순수계산!)
    [3] transfer pool wait = transfer_pool_wait (ga→ts)
    [4] transfer time = kv_transfer (ts→te)
    [5] TPOT = decode/(OSL-1)
    [total] E2EL = 공식 client (요청 전체). 전수검증: [1]~[5] 겹침 0 (903요청). (perf_metrics, warmup 제외.)
    ※주의: 여기 TTFT는 표준 TTFT(도착~첫토큰)가 아니라 사용자 정의(큐 제외 순수 prefill)."""
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

    # request lifecycle order (user spec): queuing delay → TTFT → KV-transfer queuing → transfer time → TPOT → E2EL
    # [1]~[5] = 사용자 정의 순차·절대 비겹침 단계. TTFT = "큐에서 나와 prefill 시작→첫토큰"(큐 제외 순수계산).
    #   타임라인: ca→cfs(큐) cfs→cft(TTFT) ga→ts(풀대기) ts→te(전송) ...decode(TPOT). 검증: 겹침 0.
    # E2EL = [total] (요청 전체 = 모든 단계 합), 5개와 별개로 참고용.
    panels = [
        ("[1] Queuing delay (s)", series("prefill_queue")),
        ("[2] TTFT (s) — out of queue → first token (queue-excluded)", series("prefill_compute")),
        ("[3] KV-cache transfer pool wait (s)", series("transfer_pool_wait")),
        ("[4] KV-cache transfer time (s)", series("kv_transfer")),
        ("[5] TPOT (s) — decode/(OSL-1)", tpot_series()),
        ("[total] End-to-end latency, E2EL (s)", series("e2el")),
    ]
    x = np.arange(len(labels))
    w = 0.27
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Latency decomposition — [1]→[5] are sequential NON-OVERLAPPING stages of one request; "
                 "[total]=E2EL (whole request).  mean / p50 / p99, warmup-filtered")
    for ax, (title, data) in zip(axes.flat, panels):
        m = [_num(d.get("mean")) for d in data]
        p5 = [_num(d.get("p50")) for d in data]
        p9 = [_num(d.get("p99")) for d in data]
        ax.bar(x - w, m, w, label="mean")
        ax.bar(x, p5, w, label="p50")
        ax.bar(x + w, p9, w, label="p99")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "compare_latency_decomp.png"
    fig.savefig(fname, dpi=120); plt.close(fig)
    print(f"  saved {fname}")


def plot_perside_compare(all_stats: dict, out_dir: Path) -> None:
    """[compare #2] 포인트별 per-side 동역학, 5패널:
      ① 완료율(req/s) — 1P1D라 양측 동일(conservation)
      ② 토큰 처리량(tok/s, 공식 벤치): prefill 입력=request_throughput×ISL, decode 출력=output_throughput
      ③ 동시배치  ④ transfer-pool backlog
      ⑤ per-side 이용률(%): prefill 컴퓨트 busy%(=100×rps×prefill_compute) vs decode KV풀 점유 max%.
         prefill 놀고 decode~100%면 decode 병목 → D 늘려라. 빨강 = 균형 1P:N·D.
    ※완료율 같은 건 버그 아님(conservation). 토큰비는 ISL:OSL(2:1), 비대칭은 ⑤ 이용률에서 보임."""
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

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    fig.suptitle("Per-side dynamics per point")
    w = 0.38
    # A: per-side completion rate (req/s, 공식 벤치 request rate). 1P1D라 양측 동일.
    ax = axes[0]
    pf = [g(c, p, "prefill_rps") for c, p in pts]
    dc = [g(c, p, "decode_rps") for c, p in pts]
    ax.bar(x - w / 2, pf, w, label="prefill completion rate")
    ax.bar(x + w / 2, dc, w, label="decode completion rate")
    ax.set_title("Completion rate (req/s) — equal (1P1D conservation)", fontsize=9)
    # B: per-side token throughput (공식 벤치): prefill 입력tok/s = req_throughput×ISL, decode 출력tok/s = output_throughput
    ax = axes[1]
    p_tps = [g(c, p, "prefill_input_tps_official") for c, p in pts]
    d_tps = [g(c, p, "decode_output_tps_official") for c, p in pts]
    ax.bar(x - w / 2, p_tps, w, label="prefill: input tok/s")
    ax.bar(x + w / 2, d_tps, w, label="decode: output tok/s")
    ax.set_title("Token throughput (tok/s, official bench)", fontsize=9)
    # C: concurrent batch (mean bar + max whisker)
    ax = axes[2]
    pfb = [g(c, p, "pf_bsz") for c, p in pts]
    dcb = [g(c, p, "dc_bsz") for c, p in pts]
    dcbmax = [g(c, p, "dc_bsz_max") for c, p in pts]
    ax.bar(x - w / 2, pfb, w, label="prefill batch (mean)")
    ax.bar(x + w / 2, dcb, w, label="decode batch (mean)")
    whisker(ax, x + w / 2, dcb, dcbmax)
    ax.set_title("Concurrent batch (mean bar, decode-max whisker)", fontsize=9)
    # D: backlog (mean bar + max whisker)
    ax = axes[3]
    bl = [g(c, p, "backlog") for c, p in pts]
    blmax = [g(c, p, "backlog_max") for c, p in pts]
    ax.bar(x, bl, w, label="backlog (mean)", color="tab:green")
    whisker(ax, x, bl, blmax)
    ax.set_title("Transfer-pool backlog (mean bar, max whisker)", fontsize=9)
    # E: per-side UTILIZATION (각 측이 자기 용량의 몇 % 쓰나) → 병목·노드비율. red = balanced 1P : N·D
    ax = axes[4]
    p_u = [g(c, p, "prefill_compute_util_pct") for c, p in pts]
    d_u = [g(c, p, "decode_kv_util_pct") for c, p in pts]
    ax.bar(x - w / 2, p_u, w, label="prefill: compute busy %")
    ax.bar(x + w / 2, d_u, w, label="decode: KV pool used %")
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 110)
    ratios = [g(c, p, "balanced_decode_nodes_per_prefill") for c, p in pts]
    for xi, r in zip(x, ratios):
        if r == r:                                    # NaN 아닌 것만
            ax.text(xi, 104, f"1P:{r:.1f}D", ha="center", va="bottom", fontsize=8, color="tab:red")
    ax.set_title("Utilization %: prefill busy vs decode KV  ·  red=1P:N·D", fontsize=9)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "compare_perside.png"
    fig.savefig(fname, dpi=120); plt.close(fig)
    print(f"  saved {fname}")


def _derive_capacity(s: dict, point_id: str) -> None:
    """per-side '이용률(몇 %나 쓰고 있나)' + 균형 노드비율(P:D) 파생 → s에 추가.
    1P1D에선 모든 요청이 prefill→decode를 거쳐 완료 rps가 같아짐(병목에 rate-match). 그래서
    '얼마나 빠른가(완료율)'가 아니라 '각 측이 자기 용량의 몇 %를 쓰나(이용률)'를 봐야 어디가
    병목이고 D를 몇 배 늘릴지 안다:
      prefill_compute_util_pct = 100 × 달성rps × prefill_compute  (Little ρ=λ×서비스; 컴퓨트 busy %)
      decode_kv_util_pct       = decode KV풀 점유 max %            (decode는 메모리 바운드 → KV가 곧 이용률)
      prefill_capacity_reqs_per_sec = 1 ÷ prefill_compute
      balanced_decode_nodes_per_prefill = prefill용량 ÷ 달성rps   (1P당 D 노드수; ≈ 이용률 격차)
    ※prefill 이용률 낮고 decode 이용률 ~100%면 decode 병목 → D 늘려라. context KV(held)는 prefill 일량 아님."""
    pc = s.get("prefill_compute_mean_s")
    rps = s.get("decode_rps")                               # 달성 완료율(원래 정의, 양측 동일)
    if not isinstance(rps, (int, float)):
        rps = s.get("decode_done_rps")
    # 공식 벤치 기반 per-side tps: prefill 입력tok/s = 공식 request_throughput × ISL, decode 출력tok/s = 공식 output_throughput
    isl = s.get("mean_input_len")
    if not isinstance(isl, (int, float)):
        try:
            isl, _, _ = parse_point_id(point_id)
        except Exception:
            isl = None
    if isinstance(s.get("req_s"), (int, float)) and isinstance(isl, (int, float)):
        s["prefill_input_tps_official"] = s["req_s"] * isl
    if isinstance(s.get("out_tok_s"), (int, float)):
        s["decode_output_tps_official"] = s["out_tok_s"]
    if isinstance(pc, (int, float)) and pc > 0:
        s["prefill_capacity_reqs_per_sec"] = 1.0 / pc
        if isinstance(rps, (int, float)):
            s["prefill_compute_util_pct"] = 100.0 * rps * pc            # 컴퓨트 busy %
    if isinstance(s.get("dc_kv_pct_max"), (int, float)):
        s["decode_kv_util_pct"] = s["dc_kv_pct_max"]                    # 이미 % (peak KV 점유)
    pf_cap = s.get("prefill_capacity_reqs_per_sec")
    if isinstance(pf_cap, (int, float)) and isinstance(rps, (int, float)) and rps > 0:
        s["decode_capacity_reqs_per_sec"] = rps                          # decode 포화 시 달성=용량
        s["balanced_decode_nodes_per_prefill"] = pf_cap / rps


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
            _derive_capacity(s, point_id)                                   # 순수 서비스 TPS + 균형 노드비율(P:D)
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
