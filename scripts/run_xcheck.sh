#!/usr/bin/env bash
# Run (or resume) the full Makefile workflow as a systemd user unit, detached from the terminal.
#   bash scripts/run_xcheck.sh          # start unless already running; finished steps are skipped
#   bash scripts/run_xcheck.sh status   # unit state and work/xcheck/STATUS.json
#   bash scripts/run_xcheck.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
UNIT=mfi-xcheck
WORK="${WORK:-$REPO/work/xcheck}"
mkdir -p "$WORK/logs"
case "${1:-}" in
  status) systemctl --user --no-pager status "$UNIT" 2>/dev/null | head -5 || true
          cat "$WORK/STATUS.json" 2>/dev/null || echo "no STATUS.json"; exit 0 ;;
  stop)   systemctl --user stop "$UNIT"; exit 0 ;;
esac
if systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then echo "already running ($UNIT)"; exit 0; fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
LOOP="while true; do MFI_WORK='$WORK' '$REPO/.venv/bin/python' scripts/xcheck.py refresh >/dev/null 2>&1; sleep 60; done"
systemd-run --user --unit "$UNIT" --collect --working-directory="$REPO" \
  --setenv=TMPDIR=/tmp --setenv=MPLCONFIGDIR=/tmp/mpl --setenv=HF_HOME=/tmp/hf \
  --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 --setenv=WORK="$WORK" \
  -p StandardOutput=append:"$WORK/logs/make.log" -p StandardError=append:"$WORK/logs/make.log" \
  /bin/bash -c "($LOOP) & L=\$!; make all WORK='$WORK'; rc=\$?; kill \$L; \
    MFI_WORK='$WORK' '$REPO/.venv/bin/python' scripts/xcheck.py status workflow \$([ \$rc = 0 ] && echo done || echo failed); exit \$rc"
echo "started systemd user unit $UNIT; status: $WORK/STATUS.json"
