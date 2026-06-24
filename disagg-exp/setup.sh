#!/bin/bash
# =============================================================================
# setup.sh — disagg-exp TRT-LLM 노드 부트스트랩. Idempotent — 재실행 안전.
# -----------------------------------------------------------------------------
# 권장 실행 환경: NGC 컨테이너 nvcr.io/nvidia/tensorrt-llm/release:1.2.1 **안에서**.
#   docker run --rm -it --gpus all --network host --ipc host --shm-size=8g \
#       -v "$PWD":/work -w /work/disagg-exp \
#       nvcr.io/nvidia/tensorrt-llm/release:1.2.1 bash
#   (컨테이너 안에서)  bash setup.sh
# 컨테이너엔 tensorrt_llm/CUDA/UCX/NIXL 사전설치 → 소스 빌드 안 함(변인통제: 버전 핀).
# 측정 인프라(nvidia-smi dmon / ifstat / DCGM / chrony / s5cmd)는 vLLM 1세대에서 그대로 계승.
# =============================================================================
set -euo pipefail

LOG_DIR="${EXP_LOG_DIR:-./results}"
mkdir -p "$LOG_DIR"
PY="${PYTHON:-python3}"
# 컨테이너는 보통 root → sudo 불필요/부재. 있으면 쓰고 없으면 그냥 실행.
if command -v sudo &>/dev/null; then SUDO="sudo"; else SUDO=""; fi

# ── 1. TRT-LLM 확인 (컨테이너 제공; 소스 빌드 안 함) ─────────────────────────
if ! $PY -c "import tensorrt_llm" 2>/dev/null; then
    echo "[setup] ERROR: tensorrt_llm import 실패."
    echo "        NGC 컨테이너(nvcr.io/nvidia/tensorrt-llm/release:1.2.1) 안에서 실행하세요."
    echo "        (소스 pip 빌드는 비권장 — 변인통제상 컨테이너 통째 핀 사용. CLAUDE.md 참조.)"
    exit 1
fi

# ── 2. sweep/analyze 파이썬 의존성 (컨테이너 system python에 설치) ────────────
# sweep.py = aiohttp, analyze.py = numpy(+matplotlib 플롯 선택). 나머지는 컨테이너 기본.
$PY -m pip install -q aiohttp numpy packaging huggingface_hub 2>/dev/null \
    || echo "[setup] WARN: pip install (sweep/analyze deps) 실패 — 수동 설치 필요"
$PY -m pip install -q matplotlib 2>/dev/null || true   # analyze.py --plot용(선택)

# ── 3. system tools ──────────────────────────────────────────────────────────
# ifstat: NIC 대역폭 1Hz 기록 (Prefill→Decode KV전송 바이트 관측 — inter-node 핵심)
if ! command -v ifstat &>/dev/null; then
    echo "[setup] installing ifstat ..."
    # ⚠️ apt-get update 먼저 — 핀 컨테이너는 apt 패키지목록이 비어 있어, update 없이 install하면 ifstat을 못 찾아 실패함.
    $SUDO apt-get update -qq 2>/dev/null || true
    $SUDO apt-get install -y -q ifstat 2>/dev/null || echo "[setup] WARN: apt install ifstat 실패 (apt-get update 후에도 실패 — NIC 메트릭만 부재, 핵심 측정엔 무관)"
fi

# s5cmd: 초고속 S3 백업 (aws s3 sync 폴백 있음)
if ! command -v s5cmd &>/dev/null; then
    echo "[setup] installing s5cmd ..."
    S5CMD_VER="2.2.2"
    wget -q "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VER}/s5cmd_${S5CMD_VER}_Linux-64bit.tar.gz" \
        -O /tmp/s5cmd.tar.gz \
    && tar xzf /tmp/s5cmd.tar.gz -C /tmp \
    && $SUDO mv /tmp/s5cmd /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz \
    && echo "[setup] s5cmd $(s5cmd version) installed" \
    || echo "[setup] WARN: s5cmd 설치 실패 — S3 sync 비활성"
fi

# ── 4. chrony (cross-node 시계동기 baseline — inter-node 측정에 필수) ─────────
if command -v chronyc &>/dev/null; then
    chronyc tracking > "$LOG_DIR/clock_baseline_$(hostname).txt" 2>&1 || true
    echo "[setup] chrony baseline → $LOG_DIR/clock_baseline_$(hostname).txt"
else
    echo "[setup] WARN: chronyc 없음. 호스트에 'apt install chrony' 권장(inter-node 타임스탬프 정렬용)."
fi

# ── 5. DCGM exporter (best-effort; 보통 호스트 :9400) ────────────────────────
if ! curl -sf "http://localhost:9400/metrics" | grep -q DCGM_FI 2>/dev/null; then
    if command -v dcgm-exporter &>/dev/null; then
        nohup dcgm-exporter -f /etc/dcgm-exporter/default-counters.csv \
            -a ":9400" >> "$LOG_DIR/dcgm_exporter.log" 2>&1 &
        echo "[setup] started dcgm-exporter (pid $!)"
    else
        echo "[setup] WARN: dcgm-exporter 없음. DCGM 메트릭 부재(호스트에서 기동 권장)."
    fi
fi

# ── 6. background metric collectors (1Hz GPU/NIC 기록) ───────────────────────
PIDFILE_DMON="$LOG_DIR/.pid_nvidia_dmon"
PIDFILE_IFSTAT="$LOG_DIR/.pid_ifstat"
PIDFILE_DCGM="$LOG_DIR/.pid_dcgm_loop"

_kill_pid_file() {
    local pf="$1"
    if [[ -f "$pf" ]]; then
        local pid; pid=$(cat "$pf")
        kill "$pid" 2>/dev/null || true
        rm -f "$pf"
    fi
}
_kill_pid_file "$PIDFILE_DMON"
_kill_pid_file "$PIDFILE_IFSTAT"
_kill_pid_file "$PIDFILE_DCGM"
pkill -f "nvidia-smi dmon" 2>/dev/null || true
pkill -f "ifstat -t" 2>/dev/null || true

# nvidia-smi dmon: 1Hz, Power|Util|SM Clk|Memory (컨테이너서도 --gpus all이면 동작)
nohup nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "$LOG_DIR/nvidia_smi.csv" 2>&1 &
echo $! > "$PIDFILE_DMON"
echo "[setup] nvidia-smi dmon pid=$(cat "$PIDFILE_DMON")"

# ifstat: NIC throughput 1Hz
if command -v ifstat &>/dev/null; then
    IFACE=$(ip route get 1 2>/dev/null | awk '/dev/{print $5;exit}')
    nohup ifstat -t -i "${IFACE:-eth0}" 1 \
        > "$LOG_DIR/ifstat.csv" 2>&1 &
    echo $! > "$PIDFILE_IFSTAT"
    echo "[setup] ifstat pid=$(cat "$PIDFILE_IFSTAT")"
else
    echo "[setup] WARN: ifstat 없음 — NIC 메트릭 부재."
fi

# DCGM scrape loop (PID 파일로 재실행 시 안전하게 kill)
(while true; do
    curl -sf "http://localhost:9400/metrics" \
        | grep -E "DCGM_FI_DEV_(FB_USED|GPU_UTIL|SM_OCCUPANCY|POWER_USAGE|DRAM_ACTIVE|MEM_COPY_UTILIZATION)" \
        >> "$LOG_DIR/dcgm.log" 2>/dev/null
    echo "---" >> "$LOG_DIR/dcgm.log"
    sleep 2
done) &
echo $! > "$PIDFILE_DCGM"
echo "[setup] dcgm scrape loop pid=$(cat "$PIDFILE_DCGM")"

# ── 7. validation (TRT-LLM 버전 핀 확인) ─────────────────────────────────────
$PY -c "
import tensorrt_llm
from packaging.version import Version
v = tensorrt_llm.__version__
print(f'tensorrt_llm={v}')
assert v == '1.2.1', f'expected 1.2.1, got {v} — 컨테이너 핀 확인(CLAUDE.md)'
print('OK: TRT-LLM 버전 핀 확인')
"

echo ""
echo "=== setup.sh done ==="
echo "  LOG_DIR: $LOG_DIR"
echo ""
echo "Next steps (단일 노드 예):"
echo "  # 1) 서버 기동 (ctx+gen+orchestrator)"
echo "  LABEL=T1 NUM_CTX=1 NUM_GEN=1 CTX_TP=1 CTX_PP=1 GEN_TP=1 GEN_PP=1 bash launch_trtllm.sh all"
echo "  # 2) 스윕 (다른 셸에서)"
echo "  python sweep.py --config T1 --base-url http://localhost:8000"
