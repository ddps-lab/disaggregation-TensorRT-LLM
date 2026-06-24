#!/usr/bin/env python3
"""
Workload orchestrator for disagg-exp.

부하생성·측정은 공식 `tensorrt_llm.serve.scripts.benchmark_serving`이 담당하고,
이 스크립트는 그 위의 오케스트레이션(그리드·resume·S3·metadata·per-side 스냅샷)만 한다.
즉 "공식 벤치를 최대한, 꼭 필요한 것만 커스텀" — 손수 짠 aiohttp 부하 코드는 제거됨.

Usage:
    python sweep.py --config T1 --base-url http://localhost:8000

Env overrides for the grid:
    SWEEP_PD_PAIRS="2048,128;1024,512"   # (prefill,decode) 쌍들 (';'/',' 구분)
    SWEEP_RATES=0.5,1.0,2.0              # 초당 요청 수(QPS)
    SWEEP_WARMUP_N=20  SWEEP_MEASURED_N=300

S3 sync (embedded — runs as a background thread while sweep is active):
    S3_BUCKET=hdjung-disaggregation-result   # default if --s3-bucket omitted
    S3_SYNC_INTERVAL=30                       # seconds between syncs
    --s3-bucket ""                            # disable

각 포인트가 남기는 산출물 ($EXP_LOG_DIR/<config>/, point_id=p{prefill}_d{decode}_r{rate}):
    latency_throughput_<pt>.json        # benchmark_serving --save-result (TTFT/TPOT/ITL/E2EL/throughput)
    perside_rps_<pt>.json               # per-side 카운터 스냅샷 (prefill/decode RPS 산출용)
    kv_transfer_<pt>.json               # /perf_metrics (KV전송시간)
    perside_batch_kv_backlog_<pt>.json  # 워커 /metrics 집계 (배치수·KV풀·backlog)
    timeseries_<pt>.jsonl               # 1Hz per-tick 스냅샷 (시간순 동역학 → analyze가 timeseries_<pt>.png)
"""

import argparse
import asyncio
import atexit
import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import aiohttp

import prom_scrape       # per-side(prefill/decode) RPS 스크레이퍼 — 공식이 못 주는 커스텀 조각
import metrics_sampler   # per-side 배치수·KV사용 샘플러 (워커 /metrics 폴링)


# ── 워커 /metrics 엔드포인트 (per-side 배치수·KV 샘플러용) ─────────────────────
#   launch_trtllm.sh와 동일하게 CTX_URLS/GEN_URLS env에서 워커 주소를 읽음(콤마구분 다중 가능).
#   미설정 시 localhost 기본(intra). ⚠️ inter-node면 sweep 명령에도 CTX_URLS/GEN_URLS를 줘야
#   원격 gen 워커 /metrics에 닿음 (orchestrator :8000엔 iteration stats가 없음).
def _worker_metrics_endpoints():
    import os as _os
    eps = []
    for u in [x.strip() for x in (_os.environ.get("CTX_URLS") or "localhost:8001").split(",") if x.strip()]:
        eps.append(("prefill", f"http://{u}"))
    for u in [x.strip() for x in (_os.environ.get("GEN_URLS") or "localhost:8011").split(",") if x.strip()]:
        eps.append(("decode", f"http://{u}"))
    return eps

WORKER_METRICS_EPS = _worker_metrics_endpoints()
LIVE = os.environ.get("SWEEP_LIVE", "1") == "1"   # measured 중 per-side 라이브 상태줄(stderr) 출력

# ── grid (실험 조건표) ────────────────────────────────────────────────────────
# 환경변수로 오버라이드 가능하며, (prefill,decode)×rate 교차곱이 전체 Grid를 구성합니다.

# [헬퍼] 환경변수에서 콤마 구분 텍스트를 파이썬 리스트로 변환 (예: "1.0,4.0" -> [1.0, 4.0])
def _parse_list(env_key: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_key, "")
    if raw:
        return [float(x) for x in raw.split(",")]
    return default

# PD_PAIRS: (prefill_len, decode_len) 조합. SWEEP_PD_PAIRS env로 override (smoke용 단일 포인트 등).
#   형식: "2048,128;1024,512" (쌍은 ';', 쌍 안 두 수는 ','). 미설정 시 기본 3종.
def _parse_pd_pairs(env_key: str, default: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raw = os.environ.get(env_key, "")
    if not raw:
        return default
    out: list[tuple[int, int]] = []
    for chunk in raw.split(";"):
        p, d = chunk.split(",")
        out.append((int(p), int(d)))
    return out

PD_PAIRS = _parse_pd_pairs("SWEEP_PD_PAIRS", [
    (2048, 128),
    (1024, 512),
    (128, 2048),
])
RATES        = _parse_list("SWEEP_RATES", [1.0, 2.0, 4.0])  # 초당 요청 수 (QPS)

# warmup(버림): disagg는 첫 KV전송에서 UCX 연결이 lazy 수립되므로 넉넉히 20. (서버측 CUDA graph/autotuner는
#   trtllm-serve 기동 시 자동 웜업 → 클라 웜업은 UCX·스케줄러 ramp 흡수용.) smoke 땐 SWEEP_WARMUP_N=3.
#   benchmark_serving엔 내장 warmup이 없어, measured 전에 소량 non-streaming 호출로 우리가 흡수한다.
WARMUP_N   = int(os.environ.get("SWEEP_WARMUP_N",   "20"))   # 준비운동 요청 수
MEASURED_N = int(os.environ.get("SWEEP_MEASURED_N", "300"))  # 실전 측정 요청 수

LOG_DIR = os.environ.get("EXP_LOG_DIR", "./results")  # 모든 결과 파일의 저장 경로

MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen3-4B")  # launch_trtllm.sh의 MODEL과 일치(benchmark_serving --model)

# 결과 파일명 — 자기설명적(폴더만 봐도 내용 알게). point_id = p{prefill}_d{decode}_r{rate}.
#   analyze.py의 F_* 상수와 반드시 동일하게 유지(쓰기=여기, 읽기=analyze).
F_BENCH = "latency_throughput"        # 공식 benchmark_serving: TTFT/TPOT/ITL/E2EL/throughput
F_PROM = "perside_rps"                # orchestrator ctx/gen 완료 카운터 → prefill/decode RPS
F_PERF = "kv_transfer"                # /perf_metrics: per-request KV전송 시간
F_BATCH = "perside_batch_kv_backlog"  # 워커 /metrics 집계: 배치수·KV풀·backlog
F_LIVE = "timeseries"                 # 1Hz per-tick 스냅샷(.jsonl) — 시간순 동역학


# ── 공식 부하 도구 (benchmark_serving) 호출 ────────────────────────────────────
# 부하생성·측정은 공식 tensorrt_llm.serve.scripts.benchmark_serving이 담당(손수 짠 aiohttp 대체).
#   - 토큰-id ISL 고정: --random-ids --tokenize-on-client --random-range-ratio 0 → prompt_token_ids로 정확
#   - OSL 고정: --ignore-eos --random-output-len
#   - Poisson 도착: --request-rate R --burstiness 1.0
#   - TTFT/TPOT/ITL/E2EL/throughput을 result.json에 저장(--save-result --save-detailed).
#     ⚠️ E2EL은 --percentile-metrics에 e2el을 명시해야 나옴(benchmark_serving.py:537-539).
#   내장 warmup 없음 → measured 전에 소량 non-streaming 호출로 disagg UCX cold-start 흡수.
BENCH_MODULE = "tensorrt_llm.serve.scripts.benchmark_serving"


def _split_host_port(base_url: str) -> tuple[str, int]:
    """http://host:port/... → (host, port). 포트 없으면 8000."""
    netloc = base_url.split("://", 1)[-1].split("/", 1)[0]
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        return host, int(port)
    return netloc, 8000


def build_bench_args(
    base_url: str,
    prefill_len: int,
    decode_len: int,
    rate: float,
    n: int,
    result_dir: Path | None,
    result_filename: str | None,
    streaming: bool,
) -> list[str]:
    """benchmark_serving CLI 인자 생성. result_dir 지정 시 --save-result(+--save-detailed)."""
    host, port = _split_host_port(base_url)
    args = [
        "--model", MODEL_NAME,
        "--backend", "openai",
        "--host", host,
        "--port", str(port),
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-ids",                 # 토큰-id 직접 생성(정확한 ISL)
        "--tokenize-on-client",         # 서버엔 prompt_token_ids로 전달
        "--random-input-len", str(prefill_len),
        "--random-output-len", str(decode_len),
        "--random-range-ratio", "0.0",  # 길이 분산 0(고정)
        "--ignore-eos",                 # OSL 강제
        "--num-prompts", str(n),
        "--request-rate", str(rate),
        "--burstiness", "1.0",          # 1.0 = Poisson
    ]
    if not streaming:
        args.append("--non-streaming")
    if result_dir is not None:
        args += [
            "--percentile-metrics", "ttft,tpot,itl,e2el",   # e2el 필수(아니면 latency 누락)
            "--metric-percentiles", "50,99",
            "--save-result",
            "--result-dir", str(result_dir),
            "--save-detailed",          # per-request ITL 분포 등 보존
        ]
        if result_filename:
            args += ["--result-filename", result_filename]
    return args


async def run_bench(args: list[str]) -> int:
    """benchmark_serving를 서브프로세스로 실행하고 종료코드 반환. stdout/stderr는 상속(로그에 그대로)."""
    proc = await asyncio.create_subprocess_exec(sys.executable, "-m", BENCH_MODULE, *args)
    await proc.wait()
    return proc.returncode if proc.returncode is not None else 0


# ── single sweep point (Grid 조건 1개 측정) ───────────────────────────────────
# 하나의 실험 조건(예: prefill1024 decode512 QPS1)에 대해:
#   1) warmup: 공식 benchmark_serving 소량 non-streaming 호출 (UCX cold-start 흡수, 버림)
#   2) measured: 공식 benchmark_serving 본 호출 → bench_<point>.json (집계+per-request)
#   3) per-side: measured 윈도우 전/후 orchestrator /prometheus/metrics 스냅샷 → prom_<point>.json
async def run_point(
    base_url: str,
    config: str,
    prefill_len: int,
    decode_len: int,
    rate: float,
    result_filename: str,
    out_dir: Path,
    prom_out: Path | None = None,
    batch_out: Path | None = None,
    live_out: Path | None = None,
) -> bool:
    """공식 benchmark_serving로 warmup→measured 실행. measured 결과 = out_dir/result_filename.
    prom_out 지정 시 measured 윈도우 전/후로 per-side 카운터 스냅샷(warmup 제외 = 변인통제).
    성공(rc==0 + result.json 생성) 시 True."""

    # ── 1단계: warmup (버림) — disagg UCX 첫 연결·스케줄러 ramp 흡수 ──
    #   benchmark_serving 내장 warmup 없음 → 소량 non-streaming 호출. save 안 함.
    #   짧은 고정 ISL/OSL(128/8): UCX 연결만 데우면 되므로 빠르게(decode-heavy에서도 느리지 않게).
    warmup_args = build_bench_args(base_url, 128, 8, rate,
                                   WARMUP_N, None, None, streaming=False)
    await run_bench(warmup_args)

    # ── 2단계: measured (공식 측정) — per-side 카운터로 윈도우 bracket + 배치수 샘플링 ──
    before_prom = after_prom = None   # async with 밖에서 읽으므로 미리 None
    batch_stats = None
    live_trace = None
    async with aiohttp.ClientSession() as session:
        if prom_out is not None:
            before_prom = await prom_scrape.snapshot(session, base_url)
        # per-side 배치수·KV 샘플러: measured 동안만 워커 /metrics 폴링 (warmup 제외=변인통제)
        sampler = None
        if batch_out is not None and WORKER_METRICS_EPS:
            sampler = metrics_sampler.BatchSampler(session, WORKER_METRICS_EPS, interval=1.0,
                                                   orchestrator_url=base_url, live=LIVE)
            sampler.start()
        measured_args = build_bench_args(base_url, prefill_len, decode_len, rate,
                                         MEASURED_N, out_dir, result_filename, streaming=True)
        rc = await run_bench(measured_args)
        if sampler is not None:
            batch_stats = await sampler.stop()   # 폴링 종료 + per-side 평균/최대 집계
            live_trace = sampler.trace            # 매초 스냅샷(라이브 기록)
        if prom_out is not None:
            after_prom = await prom_scrape.snapshot(session, base_url)

    # ── 3단계: per-side 스냅샷 저장 (analyze가 RPS=(after-before)/window_s 계산) ──
    if prom_out is not None and before_prom is not None and after_prom is not None:
        window_s = after_prom.get("ts", 0) - before_prom.get("ts", 0)
        with open(prom_out, "w") as f:
            json.dump({"before": before_prom, "after": after_prom, "window_s": window_s}, f, indent=2)
    if batch_out is not None and batch_stats:
        with open(batch_out, "w") as f:
            json.dump(batch_stats, f, indent=2)
    if live_out is not None and live_trace:
        with open(live_out, "w") as f:   # jsonl: 한 줄 = 한 tick 스냅샷(라이브 기록)
            for row in live_trace:
                f.write(json.dumps(row) + "\n")

    # 성공판정: benchmark_serving 종료코드 0 + result.json 생성
    result_path = out_dir / result_filename
    ok = (rc == 0) and result_path.exists()
    if not ok:
        print(f"  [bench] FAILED rc={rc} result_exists={result_path.exists()}", flush=True)
    return ok


# ── S3 sync (백그라운드 자동 백업) ─────────────────────────────────────────────
# 실험 도중 서버가 꺼져도 데이터가 살아남도록, 30초마다 ./results/ → S3로 복사합니다.
# s5cmd(빠름)이 있으면 쓰고, 없으면 aws cli로 폴백합니다.
# --s3-bucket "" 로 비활성화 가능합니다.
class S3Syncer:

    SYNC_TIMEOUT_S = 300  # 1회 sync 최대 대기 시간

    def __init__(self, bucket: str, log_dir: str, config: str, interval: int = 30):
        self.bucket = bucket
        self.log_dir = log_dir
        self.interval = interval
        # S3 저장 경로: s3://버킷/raw/custom/20260518/hostname/configA1/
        # "custom" prefix → 우리 하네스(sweep + benchmark_serving) 산출물 경로 구분용.
        # config 한 단계 더 → 같은 호스트에서 여러 config 결과 보관해도 안 섞임.
        date = _dt.datetime.utcnow().strftime("%Y%m%d")
        host = socket.gethostname()
        self.dest = f"s3://{bucket}/raw/custom/{date}/{host}/{config}/"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._final_done = False
        self._lock = threading.Lock()
        self._cmd = self._pick_cmd()  # s5cmd 또는 aws cli 자동 탐색
        self._log_path = Path(log_dir) / "s3_sync.log"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pick_cmd() -> list[str] | None:
        if shutil.which("s5cmd"):
            return ["s5cmd", "sync"]
        if shutil.which("aws"):
            return ["aws", "s3", "sync"]
        return None

    def _log(self, msg: str) -> None:
        ts = _dt.datetime.utcnow().isoformat(timespec="seconds")
        line = f"[{ts}] {msg}\n"
        try:
            with open(self._log_path, "a") as f:
                f.write(line)
        except Exception:
            pass
        print(f"[s3_sync] {msg}", flush=True)

    def _sync_once(self) -> None:
        """로컬 폴더 전체를 S3로 1회 복사 (이동이 아님 — 로컬 원본은 유지)"""
        if not self._cmd:
            return
        argv = self._cmd + [f"{self.log_dir}/", self.dest]
        try:
            with open(self._log_path, "a") as logf:
                subprocess.run(
                    argv,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=self.SYNC_TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:
            self._log(f"sync timed out after {self.SYNC_TIMEOUT_S}s")
        except Exception as exc:
            self._log(f"sync error: {exc}")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._sync_once()
            # Interruptible sleep: stop_event triggers immediate wakeup.
            self._stop_event.wait(timeout=self.interval)

    def start(self) -> None:
        if not self.bucket:
            print("[s3_sync] disabled (no bucket)", flush=True)
            return
        if not self._cmd:
            self._log("WARN: neither s5cmd nor aws cli found — S3 sync DISABLED")
            return
        self._log(f"starting (cmd={self._cmd[0]}, every {self.interval}s) → {self.dest}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="s3-sync")
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        # Guard against double-invocation (manual stop + atexit).
        with self._lock:
            if self._final_done:
                return
            self._final_done = True

        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        # Final sync regardless — captures whatever the loop missed.
        self._log("final sync ...")
        self._sync_once()
        self._log("stopped")


# ── health check ──────────────────────────────────────────────────────────────
# [헬스 체크 함수] 서버가 완전히 켜져서 트래픽을 받을 준비가 될 때까지 기다립니다.
async def wait_for_health(base_url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    print(f"Waiting for {base_url}/v1/completions ...", flush=True)
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.time() < deadline:
            try:
                async with session.post(
                    f"{base_url}/v1/completions",
                    json={"model": MODEL_NAME, "prompt": "hi", "max_tokens": 1},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        print("  server ready.", flush=True)
                        return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"Server at {base_url} not healthy after {timeout_s}s")


# ── perf metrics 수집 (KV전송시간 등) ──────────────────────────────────────────
# 포인트 측정 종료 후 orchestrator /perf_metrics를 1회 폴링해 원본 저장.
#   - 응답: List[Dict] (FIFO, 요청 ID 없음). KV전송시간 = gen_perf_metrics[.perf_metrics].timing_metrics의
#     kv_cache_transfer_end-start(초, kv_cache_size>0일 때만). analyze.py가 None-safe 파싱(perf_metrics 래퍼 유무 둘 다).
#   - 활성: disagg YAML perf_metrics_max_requests>0 + 워커 extra YAML return_perf_metrics:true.
#   - 비활성/엔드포인트 없음이면 조용히 skip — 측정엔 영향 없음(변인통제: 측정 후 1회만 폴링).
async def fetch_perf_metrics(base_url: str, out_path: Path) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/perf_metrics",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    print(f"  [perf] /perf_metrics HTTP {resp.status} — skip", flush=True)
                    return
                data = await resp.json()
    except Exception as exc:
        print(f"  [perf] fetch skip: {exc}", flush=True)
        return
    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)   # 사람이 IDE에서 보기 쉽게 들여쓰기
        recs = data if isinstance(data, list) else []

        def _has_kv(e: dict) -> bool:
            gen = (e.get("gen_perf_metrics") or {}) if isinstance(e, dict) else {}
            gen = gen.get("perf_metrics") or gen   # perf_metrics 래퍼 있으면 벗김(둘 다 대응)
            return (gen.get("timing_metrics") or {}).get("kv_cache_transfer_start") is not None

        n_kv = sum(1 for e in recs if _has_kv(e))
        print(f"  [perf] {len(recs)} records ({n_kv} w/ KV transfer) → {out_path.name}", flush=True)
    except Exception as exc:
        print(f"  [perf] save failed: {exc}", flush=True)


# ── main (전체 실험 오케스트레이터) ────────────────────────────────────────────
# 실행 순서: 영수증 생성 → S3 백업 시작 → 서버 대기 → Grid 조건표 생성 → 순차 실행
async def main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    config = args.config
    out_dir = Path(LOG_DIR) / config  # 예: ./results/T1/
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 실험 영수증 생성 (나중에 "이 폴더가 뭐였지?" 할 때 보는 파일) ──
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        # TRT-LLM PD 토폴로지를 환경변수에서 기록 (launch_trtllm.sh와 동일 변수).
        # 하드코딩 맵 대신 실제 실행값을 그대로 보존 → P/D의 (TP,PP)·xPyD·placement 분리.
        ctx_tp = int(os.environ.get("CTX_TP", "1"))
        ctx_pp = int(os.environ.get("CTX_PP", "1"))
        gen_tp = int(os.environ.get("GEN_TP", "1"))
        gen_pp = int(os.environ.get("GEN_PP", "1"))
        num_ctx = int(os.environ.get("NUM_CTX", "1"))
        num_gen = int(os.environ.get("NUM_GEN", "1"))
        placement = os.environ.get("PLACEMENT", "intra")        # intra | inter
        cache_backend = os.environ.get("CACHE_BACKEND", "UCX")  # KV 전송 백엔드
        meta = {
            "config": config,
            "framework": "tensorrt-llm",
            "version": "1.2.1",
            "model": MODEL_NAME,
            "load_tool": "benchmark_serving",   # 부하·측정 = 공식 도구 (sweep은 오케스트레이션)
            "context": {"num_instances": num_ctx, "tp": ctx_tp, "pp": ctx_pp},
            "generation": {"num_instances": num_gen, "tp": gen_tp, "pp": gen_pp},
            "placement": placement,
            "cache_transceiver_backend": cache_backend,
            "start_time": _dt.datetime.utcnow().isoformat(),
            "description": (
                f"TRT-LLM PD-disagg benchmark — {config} "
                f"(ctx {num_ctx}x tp{ctx_tp}pp{ctx_pp} / gen {num_gen}x tp{gen_tp}pp{gen_pp}, {placement})"
            ),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── S3 백업 시작 (실험 도중 서버가 죽어도 데이터 보존) ──
    syncer = S3Syncer(
        bucket=args.s3_bucket or "",
        log_dir=LOG_DIR,
        config=config,
        interval=int(os.environ.get("S3_SYNC_INTERVAL", "30")),
    )
    syncer.start()

    # ── 서버가 켜질 때까지 대기 ──
    await wait_for_health(base_url)

    # ── Grid 조건표 생성 ──
    # 정의된 (Prefill, Decode) 조합과 Rate의 교차 조합 생성
    points: list[tuple[int, int, float]] = []
    for pl, dl in PD_PAIRS:
        for r in RATES:
            points.append((pl, dl, r))

    print(f"Grid: {len(points)} points × (warmup={WARMUP_N} + measured={MEASURED_N}), load=benchmark_serving", flush=True)

    done = 0
    skipped = 0
    for prefill_len, decode_len, rate in points:
        point_id = f"p{prefill_len}_d{decode_len}_r{rate}"  # 파일명 = 실험 조건
        bench_file = f"{F_BENCH}_{point_id}.json"            # benchmark_serving --save-result 결과
        marker_done   = out_dir / f".done_{point_id}"        # 완료 도장 (재실행 시 스킵)
        marker_failed = out_dir / f".failed_{point_id}"      # 실패 도장

        # 이미 완료된 조건은 건너뜀 → 중간에 끊겨도 이어서 실행 가능
        if marker_done.exists():
            skipped += 1
            continue

        print(f"[{done+1}/{len(points)}] prefill={prefill_len} decode={decode_len} rate={rate} ...", flush=True)

        try:
            ok = await run_point(base_url, config, prefill_len, decode_len, rate,
                                 result_filename=bench_file, out_dir=out_dir,
                                 prom_out=out_dir / f"{F_PROM}_{point_id}.json",
                                 batch_out=out_dir / f"{F_BATCH}_{point_id}.json",
                                 live_out=out_dir / f"{F_LIVE}_{point_id}.jsonl")
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            marker_failed.touch()
            continue

        if ok:
            marker_done.touch()
            marker_failed.unlink(missing_ok=True)
            # KV전송시간 등 per-request perf 수집 (측정 종료 후 1회). 비활성이면 조용히 skip.
            await fetch_perf_metrics(base_url, out_dir / f"{F_PERF}_{point_id}.json")
        else:
            marker_failed.touch()

        done += 1

    print(f"\nDone. {done} run, {skipped} skipped (already done).", flush=True)

    # Explicit stop so the final sync runs before process exit messages clear
    # the terminal. atexit would also call it, but the guard prevents double-runs.
    syncer.stop()


# ── 진입점: python sweep.py --config T1 --base-url http://localhost:8000 ──────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config 라벨 (예: T1|T2|T3|T4). analyze.py COST_PER_HR 키와 일치시킬 것")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("S3_BUCKET", "hdjung-disaggregation-result"),
        help='S3 bucket for background sync. Pass "" to disable.',
    )
    args = parser.parse_args()
    asyncio.run(main(args))  # 비동기 이벤트 루프 시작 → main() 실행
