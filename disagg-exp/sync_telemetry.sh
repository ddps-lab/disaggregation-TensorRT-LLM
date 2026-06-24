#!/bin/bash
# =============================================================================
# sync_telemetry.sh — sweep를 안 돌리는 노드(=decode 노드)의 텔레메트리를 S3로 자동 백업.
# -----------------------------------------------------------------------------
# 배경: S3 백업(S3Syncer)은 sweep.py가 도는 노드(node1=prefill+orchestrator)에서만 돈다.
#       decode 노드(node2)는 sweep를 안 돌려 자기 results/(nvidia_smi.csv·ifstat.csv·dcgm.log)가
#       S3로 안 올라간다. 이 스크립트가 그 한 줄을 메운다 — node2에서 백그라운드로 돌리면
#       node1과 "같은 날짜/같은 config, 다른 hostname" 경로로 나란히 쌓인다(충돌 없음).
#
#   왜 decode 텔레메트리?  dc_kv%(소프트웨어 KV 점유율)만으론 "디코드가 메모리 바운드라
#   막힌다"를 입증 못 한다. node2 nvidia-smi(SM util·메모리·전력) = 그 물증, node2 ifstat = KV
#   전송의 수신(RX)측(node1은 송신 TX만 봄). (자세한 근거: README / CLAUDE.md)
#
# 경로 스키마는 sweep.py의 S3Syncer와 동일:
#       s3://$S3_BUCKET/raw/custom/<YYYYMMDD>/<hostname>/<CONFIG>/
#
# 사용 (decode 노드, 컨테이너 안에서 — setup.sh로 수집기가 이미 돌고 있어야 함):
#       CONFIG=T1 bash sync_telemetry.sh start    # 백그라운드 시작(pid 출력)
#       bash sync_telemetry.sh stop               # 중지 + 마지막 1회 sync
#
# ENV:
#   S3_BUCKET          기본 hdjung-disaggregation-result   ("" 면 비활성)
#   CONFIG             S3 하위 폴더 = node1의 --config 와 같게 (기본 smoke)
#   EXP_LOG_DIR        올릴 폴더 (기본 ./results)
#   S3_SYNC_INTERVAL   sync 간격(초, 기본 30)
# =============================================================================
set -euo pipefail

# colon 없는 기본값 — UNSET이면 기본 버킷, 빈 문자열("")이면 그대로 빈값=비활성(S3Syncer의 `or ""`와 동일 의도).
S3_BUCKET="${S3_BUCKET-hdjung-disaggregation-result}"
CONFIG="${CONFIG:-smoke}"
LOG_DIR="${EXP_LOG_DIR:-./results}"
INTERVAL="${S3_SYNC_INTERVAL:-30}"
ACTION="${1:-start}"

mkdir -p "$LOG_DIR"
PIDFILE="$LOG_DIR/.pid_telemetry_sync"
SYNC_LOG="$LOG_DIR/s3_sync.log"

# s5cmd(빠름) 우선, 없으면 aws cli 폴백 — 둘 다 `<cmd> sync SRC DEST` 형태(S3Syncer와 동일).
_pick_cmd() {
    if command -v s5cmd &>/dev/null; then echo "s5cmd sync"; return; fi
    if command -v aws   &>/dev/null; then echo "aws s3 sync"; return; fi
    echo ""
}

_stop() {
    if [[ -f "$PIDFILE" ]]; then
        local pid; pid=$(cat "$PIDFILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "[telemetry_sync] stopped (pid $pid)"
    else
        echo "[telemetry_sync] not running (no pidfile)"
    fi
}

case "$ACTION" in
stop)
    _stop
    # 마지막 1회 — 루프가 놓친 마지막 몇 초 telemetry까지 확실히 올림.
    CMD="$(_pick_cmd)"
    if [[ -n "$CMD" && -n "$S3_BUCKET" ]]; then
        DATE="$(date -u +%Y%m%d)"; HOST="$(hostname)"
        DEST="s3://$S3_BUCKET/raw/custom/$DATE/$HOST/$CONFIG/"
        echo "[telemetry_sync] final sync → $DEST"
        $CMD "$LOG_DIR/" "$DEST" >> "$SYNC_LOG" 2>&1 || true
    fi
    exit 0
    ;;
start) ;;
*)
    echo "usage: [CONFIG=..] bash sync_telemetry.sh {start|stop}" >&2
    exit 2
    ;;
esac

# ── start ────────────────────────────────────────────────────────────────────
if [[ -z "$S3_BUCKET" ]]; then
    echo "[telemetry_sync] disabled (S3_BUCKET 비어 있음)"; exit 0
fi
CMD="$(_pick_cmd)"
if [[ -z "$CMD" ]]; then
    echo "[telemetry_sync] WARN: s5cmd/aws 둘 다 없음 — S3 sync 비활성 (setup.sh로 s5cmd 설치 권장)"; exit 0
fi

# 이전 인스턴스가 떠 있으면 정리(idempotent — 재실행 안전).
if [[ -f "$PIDFILE" ]]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi

DATE="$(date -u +%Y%m%d)"; HOST="$(hostname)"
DEST="s3://$S3_BUCKET/raw/custom/$DATE/$HOST/$CONFIG/"

# 백그라운드 루프: INTERVAL초마다 results/ 전체를 S3로 복사(이동 아님 — 로컬 원본 유지).
nohup bash -c '
    dest="'"$DEST"'"; src="'"$LOG_DIR"'/"; interval="'"$INTERVAL"'"; cmd="'"$CMD"'"; log="'"$SYNC_LOG"'"
    echo "[$(date -u +%FT%TZ)] telemetry_sync start (cmd=${cmd%% *}, every ${interval}s) -> $dest" >> "$log"
    while true; do
        $cmd "$src" "$dest" >> "$log" 2>&1 || true
        sleep "$interval"
    done
' >> "$SYNC_LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "[telemetry_sync] started (pid $(cat "$PIDFILE"), every ${INTERVAL}s) → $DEST"
echo "[telemetry_sync] 중지: bash sync_telemetry.sh stop"
