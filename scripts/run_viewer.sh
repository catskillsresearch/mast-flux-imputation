#!/usr/bin/env bash
# Run the viewer (app/fec_site) detached as a systemd user unit, so it survives the terminal/app.
#   bash scripts/run_viewer.sh [start] [PORT]   # default port 8770, http://127.0.0.1:PORT/
#   bash scripts/run_viewer.sh status
#   bash scripts/run_viewer.sh stop
#   bash scripts/run_viewer.sh restart [PORT]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
UNIT=mfi-viewer
CMD=${1:-start}
PORT=${2:-8770}
case "$CMD" in
  status) systemctl --user --no-pager status "$UNIT" | head -8; exit 0 ;;
  stop) systemctl --user stop "$UNIT" && echo "stopped $UNIT"; exit 0 ;;
  restart) systemctl --user stop "$UNIT" 2>/dev/null || true ;;
  start) if systemctl --user is-active --quiet "$UNIT"; then echo "already running ($UNIT)"; exit 0; fi ;;
  *) echo "usage: $0 [start|stop|status|restart] [PORT]"; exit 2 ;;
esac
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
mkdir -p outputs
systemd-run --user --unit "$UNIT" --collect --working-directory="$REPO" \
  --setenv=CATSKILLS_FUSION_LAB=/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge \
  --setenv=TMPDIR=/tmp --setenv=MPLCONFIGDIR=/tmp/mpl --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=OMP_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 \
  -p StandardOutput=append:"$REPO/outputs/viewer.log" -p StandardError=append:"$REPO/outputs/viewer.log" \
  "$REPO/.venv/bin/python" -m app.fec_site --port "$PORT"
echo "started $UNIT on http://127.0.0.1:$PORT/"
