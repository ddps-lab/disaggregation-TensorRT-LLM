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
#
# ⚠️ 진행상황 가시성(사용자 요구): pip/apt/wget 의 실제 작업은 -q/-qq/2>/dev/null 로 숨기지 않는다.
#    → 다운로드·설치·에러가 화면에 전부 출력된다. 각 단계는 [setup] N/7 배너로 어디까지 왔는지 표시.
#    (curl :9400 헬스 probe, kill/pkill 정리, command -v 존재확인 같은 "진행상황 아닌" 노이즈만 조용히 둔다.)
# =============================================================================
set -euo pipefail

LOG_DIR="${EXP_LOG_DIR:-./results}"
TELE_DIR="$LOG_DIR/telemetry"   # 노드 텔레메트리(nvidia-smi·ifstat·dcgm·clock·pid)는 여기로 (results 루트 정리)
mkdir -p "$LOG_DIR" "$TELE_DIR"
PY="${PYTHON:-python3}"
# 컨테이너는 보통 root → sudo 불필요/부재. 있으면 쓰고 없으면 그냥 실행.
if command -v sudo &>/dev/null; then SUDO="sudo"; else SUDO=""; fi

echo "[setup] ===== START — 모든 단계 진행상황 출력(quiet/2>/dev/null 제거). LOG_DIR=$LOG_DIR ====="

# ── 1. TRT-LLM 확인 (컨테이너 제공; 소스 빌드 안 함) ─────────────────────────
echo "[setup] 1/7: tensorrt_llm import 확인 ..."
if ! $PY -c "import tensorrt_llm"; then   # 실패하면 import 에러 그대로 보이게(2>/dev/null 제거)
    echo "[setup] ERROR: tensorrt_llm import 실패."
    echo "        NGC 컨테이너(nvcr.io/nvidia/tensorrt-llm/release:1.2.1) 안에서 실행하세요."
    echo "        (소스 pip 빌드는 비권장 — 변인통제상 컨테이너 통째 핀 사용. CLAUDE.md 참조.)"
    exit 1
fi

# ── 2. sweep/analyze 파이썬 의존성 (컨테이너 system python에 설치) ────────────
# sweep.py = aiohttp, analyze.py = numpy(+matplotlib 플롯 선택). 나머지는 컨테이너 기본.
# 진행상황 다 보이게: pip -q / 2>/dev/null 안 씀(다운로드·에러 전부 출력).
echo "[setup] 2/7: pip install (aiohttp numpy packaging huggingface_hub) — 다운로드 진행 표시 ..."
$PY -m pip install aiohttp numpy packaging huggingface_hub \
    || echo "[setup] WARN: pip install (sweep/analyze deps) 실패 — 수동 설치 필요"
echo "[setup] 2/7: pip install matplotlib (analyze --plot용, 선택) ..."
$PY -m pip install matplotlib || echo "[setup] WARN: matplotlib 설치 실패(플롯만 영향)"

# ── 3. system tools ──────────────────────────────────────────────────────────
echo "[setup] 3/7: system tools (ifstat, s5cmd) ..."
# ifstat: NIC 대역폭 1Hz 기록 (Prefill→Decode KV전송 바이트 관측 — inter-node 핵심)
if ! command -v ifstat &>/dev/null; then
    echo "[setup]   installing ifstat (apt — 진행 표시) ..."
    # ⚠️ apt-get update 먼저 — 핀 컨테이너는 apt 패키지목록이 비어 있어, update 없이 install하면 ifstat을 못 찾아 실패함.
    $SUDO apt-get update || echo "[setup]   WARN: apt-get update 실패(계속 진행)"
    $SUDO apt-get install -y ifstat || echo "[setup]   WARN: apt install ifstat 실패 (NIC 메트릭만 부재, 핵심 측정엔 무관)"
fi

# s5cmd: 초고속 S3 백업 (aws s3 sync 폴백 있음)
if ! command -v s5cmd &>/dev/null; then
    echo "[setup]   installing s5cmd (wget 다운로드 — 진행률 표시) ..."
    S5CMD_VER="2.2.2"
    wget --progress=bar:force "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VER}/s5cmd_${S5CMD_VER}_Linux-64bit.tar.gz" \
        -O /tmp/s5cmd.tar.gz \
    && tar xzf /tmp/s5cmd.tar.gz -C /tmp \
    && $SUDO mv /tmp/s5cmd /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz \
    && echo "[setup]   s5cmd $(s5cmd version) installed" \
    || echo "[setup]   WARN: s5cmd 설치 실패 — S3 sync 비활성"
fi

# ── 4. chrony (cross-node 시계동기 baseline — inter-node 타임스탬프 정렬 기록용) ──
# 컨테이너는 호스트 커널 시계를 공유한다(안에서 시계 변경 불가). 그래서 여기선 "현재 얼마나
# 잘 맞는지"를 baseline으로만 기록한다. AWS는 Amazon Time Sync(169.254.169.123)로 호스트가 이미
# 동기화됨. 우선순위: chronyc tracking(호스트 chronyd 접근 시) → chronyd -Q(시계 안 건드리고 NTP
# offset 1회 측정, 컨테이너서 동작) → date. 어느 쪽이든 baseline 파일은 항상 채워진다.
echo "[setup] 4/7: chrony 시계 baseline ..."
if ! command -v chronyc &>/dev/null && ! command -v chronyd &>/dev/null; then
    echo "[setup]   installing chrony (apt — 진행 표시) ..."
    $SUDO apt-get update || echo "[setup]   WARN: apt-get update 실패(계속)"   # ifstat에서 안 돌았을 수 있어 한 번 더(idempotent)
    $SUDO apt-get install -y chrony || echo "[setup]   WARN: apt install chrony 실패 — date 기반 baseline으로 폴백"
fi
CLOCK_BASE="$TELE_DIR/clock_baseline_$(hostname).txt"
# chronyd는 /usr/sbin에 설치돼 root PATH에 없을 수 있어 명시 경로도 확인.
CHRONYD_BIN="$(command -v chronyd 2>/dev/null || true)"
[[ -z "$CHRONYD_BIN" && -x /usr/sbin/chronyd ]] && CHRONYD_BIN=/usr/sbin/chronyd
{
    echo "# clock baseline @ $(date -u +%FT%TZ) host=$(hostname)"
    if command -v chronyc &>/dev/null && chronyc tracking 2>/dev/null; then
        echo "# (source: chronyc tracking — 호스트 chronyd 접근됨)"
    elif [[ -n "$CHRONYD_BIN" ]]; then
        echo "# (source: chronyd -Q one-shot NTP query → Amazon Time Sync; 시계는 안 건드림)"
        timeout 20 $SUDO "$CHRONYD_BIN" -Q 'server 169.254.169.123 iburst' 2>&1 \
          || timeout 20 "$CHRONYD_BIN" -Q 'server pool.ntp.org iburst' 2>&1 \
          || echo "WARN: chronyd -Q 실패 — date만 기록"
    else
        echo "WARN: chrony 미설치 — date만 기록(호스트가 Amazon Time Sync로 동기화됨을 전제)"
    fi
    echo "# date -u(기록완료): $(date -u +%FT%T.%NZ)"
} > "$CLOCK_BASE" 2>&1
echo "[setup] clock baseline → $CLOCK_BASE"

# ── 5. DCGM exporter (선택 — nvidia-smi dmon이 GPU util/mem-bw/power/clk/PCIe를 이미 1Hz로 커버) ──
# DCGM은 보강용(SM_OCCUPANCY/DRAM_ACTIVE 등). 보통 호스트나 별도 dcgm-exporter 컨테이너가 :9400로 노출.
# 컨테이너 안에 바이너리가 있으면 best-effort로 띄워봄(대개 없음 → 정상, §6에서 건너뜀).
echo "[setup] 5/7: DCGM exporter 확인(선택 — 보통 컨테이너엔 없어 건너뜀) ..."
if ! curl -sf "http://localhost:9400/metrics" 2>/dev/null | grep -q DCGM_FI; then   # :9400 probe — 없으면 connection-refused 스팸이라 조용히
    if command -v dcgm-exporter &>/dev/null; then
        nohup dcgm-exporter -f /etc/dcgm-exporter/default-counters.csv \
            -a ":9400" >> "$TELE_DIR/dcgm_exporter.log" 2>&1 &
        echo "[setup] started dcgm-exporter (pid $!)"
    fi
fi

# ── 6. background metric collectors (1Hz GPU/NIC 기록) ───────────────────────
echo "[setup] 6/7: 백그라운드 수집기 기동 (nvidia-smi dmon, ifstat) ..."
PIDFILE_DMON="$TELE_DIR/.pid_nvidia_dmon"
PIDFILE_IFSTAT="$TELE_DIR/.pid_ifstat"
PIDFILE_DCGM="$TELE_DIR/.pid_dcgm_loop"

_kill_pid_file() {
    local pf="$1"
    if [[ -f "$pf" ]]; then
        local pid; pid=$(cat "$pf")
        kill "$pid" 2>/dev/null || true   # 정리(cleanup) — 진행상황 아님
        rm -f "$pf"
    fi
}
_kill_pid_file "$PIDFILE_DMON"
_kill_pid_file "$PIDFILE_IFSTAT"
_kill_pid_file "$PIDFILE_DCGM"
pkill -f "nvidia-smi dmon" 2>/dev/null || true   # 정리 — 진행상황 아님
pkill -f "ifstat -t" 2>/dev/null || true

# nvidia-smi dmon: 1Hz, Power|Util|SM Clk|Memory (컨테이너서도 --gpus all이면 동작)
nohup nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "$TELE_DIR/nvidia_smi.csv" 2>&1 &
echo $! > "$PIDFILE_DMON"
echo "[setup]   nvidia-smi dmon pid=$(cat "$PIDFILE_DMON") → $TELE_DIR/nvidia_smi.csv"

# ifstat: NIC throughput 1Hz
if command -v ifstat &>/dev/null; then
    IFACE=$(ip route get 1 2>/dev/null | awk '/dev/{print $5;exit}')   # 인터페이스 탐지 probe
    nohup ifstat -t -i "${IFACE:-eth0}" 1 \
        > "$TELE_DIR/ifstat.csv" 2>&1 &
    echo $! > "$PIDFILE_IFSTAT"
    echo "[setup]   ifstat pid=$(cat "$PIDFILE_IFSTAT") iface=${IFACE:-eth0} → $TELE_DIR/ifstat.csv"
else
    echo "[setup]   WARN: ifstat 없음 — NIC 메트릭 부재."
fi

# DCGM scrape loop — :9400가 실제로 살아있을 때만. 없으면 건너뜀(없는데 도는 빈 루프 방지;
# 핵심 GPU 메트릭은 위 nvidia-smi dmon이 이미 기록). PID 파일로 재실행 시 안전하게 kill.
if curl -sf "http://localhost:9400/metrics" 2>/dev/null | grep -q DCGM_FI; then   # :9400 probe
    (while true; do
        curl -sf "http://localhost:9400/metrics" \
            | grep -E "DCGM_FI_DEV_(FB_USED|GPU_UTIL|SM_OCCUPANCY|POWER_USAGE|DRAM_ACTIVE|MEM_COPY_UTILIZATION)" \
            >> "$TELE_DIR/dcgm.log" 2>/dev/null
        echo "---" >> "$TELE_DIR/dcgm.log"
        sleep 2
    done) &
    echo $! > "$PIDFILE_DCGM"
    echo "[setup]   dcgm scrape loop pid=$(cat "$PIDFILE_DCGM") (:9400 live)"
else
    echo "[setup]   DCGM :9400 없음 → scrape 건너뜀 (선택사항; nvidia-smi dmon이 util/mem-bw/power/clk/PCIe 커버)"
fi

# ── 7. validation (TRT-LLM 버전 핀 확인) ─────────────────────────────────────
echo "[setup] 7/7: TRT-LLM 버전 핀(1.2.1) 검증 ..."
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
