#!/usr/bin/env bash
set -e

cd /sdc/huoyongzhen/AF-CLIP-clean
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate multiads
fi
mkdir -p logs/prcv_laclip_auto reports/prcv_laclip_auto
tmux new -d -s laclip_all_ablations "PYTHONUNBUFFERED=1 /home/huoyongzhen/.conda/envs/multiads/bin/python scripts/prcv_laclip_runs/run_all_prcv_experiments.py 2>&1 | tee logs/prcv_laclip_auto/scheduler.log"
echo "Started tmux session: laclip_all_ablations"
echo "Attach with: tmux attach -t laclip_all_ablations"
echo "Watch scheduler log: tail -f logs/prcv_laclip_auto/scheduler.log"
echo "Watch state: watch -n 30 'cat reports/prcv_laclip_auto/scheduler_state.md'"
