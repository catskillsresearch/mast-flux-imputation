#!/usr/bin/env bash
# Launch (or resume) the E6 driver detached from the terminal / app session (systemd user unit).
#   bash scripts/run_e6.sh          # start unless already running; completed stages are skipped
#   bash scripts/run_e6.sh status   # unit state and outputs/experiments/E6/STATUS.json
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
UNIT=mfi-e6
mkdir -p outputs/experiments/E6
if [[ "${1:-}" == "status" ]]; then
  systemctl --user --no-pager status "$UNIT" 2>/dev/null | head -5 || true
  cat outputs/experiments/E6/STATUS.json 2>/dev/null || echo "no STATUS.json"
  exit 0
fi
if systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then
  echo "already running ($UNIT)"; exit 0
fi
if pgrep -f "experiments.driver_e6" >/dev/null; then
  echo "a driver process is already running:"; pgrep -af "experiments.driver_e6"; exit 0
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
ENVV=(CATSKILLS_FUSION_LAB=/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge TMPDIR=/tmp HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
      MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1)
if command -v systemd-run >/dev/null && systemctl --user is-system-running >/dev/null 2>&1; then
  args=(); for e in "${ENVV[@]}"; do args+=(--setenv="$e"); done
  systemd-run --user --unit "$UNIT" --collect --working-directory="$REPO" "${args[@]}" \
    -p StandardOutput=append:"$REPO/outputs/experiments/E6/driver.log" \
    -p StandardError=append:"$REPO/outputs/experiments/E6/driver.log" \
    "$REPO/.venv/bin/python" -m experiments.driver_e6
  echo "started systemd user unit $UNIT"
else
  env -u MFI_OUTPUT_DIR "${ENVV[@]}" setsid nohup "$REPO/.venv/bin/python" -m experiments.driver_e6 \
    >>outputs/experiments/E6/driver.log 2>&1 < /dev/null &
  echo "started with setsid, pid $!"
fi
