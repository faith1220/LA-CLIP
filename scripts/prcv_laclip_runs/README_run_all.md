# PRCV Auto Scheduler

## Files

- `run_all_prcv_experiments.py`: the main queue scheduler
- `launch_all_prcv_experiments.sh`: starts the scheduler inside `tmux`
- `export_visualizations.py`: exports the final visualization panels after numeric experiments finish

## Start

```bash
cd /sdc/huoyongzhen/AF-CLIP-clean
bash scripts/prcv_laclip_runs/launch_all_prcv_experiments.sh
```

This creates the `tmux` session `laclip_all_ablations`.

## Monitor

```bash
tmux ls
tail -f logs/prcv_laclip_auto/scheduler.log
cat reports/prcv_laclip_auto/scheduler_state.md
cat reports/prcv_laclip_auto/experiment_summary.md
```

## Stop

```bash
tmux kill-session -t laclip_all_ablations
```

The scheduler traps `SIGINT` and `SIGTERM`, writes its latest state, and terminates experiments launched by the scheduler process.

## Output Paths

- Scheduler log: `logs/prcv_laclip_auto/scheduler.log`
- Per-experiment logs: `logs/prcv_laclip_auto/<experiment>.log`
- State JSON: `reports/prcv_laclip_auto/scheduler_state.json`
- State Markdown: `reports/prcv_laclip_auto/scheduler_state.md`
- Summary CSV: `reports/prcv_laclip_auto/experiment_summary.csv`
- Summary Markdown: `reports/prcv_laclip_auto/experiment_summary.md`
- Commands: `reports/prcv_laclip_auto/commands/<experiment>.sh`
- Visualizations: `reports/prcv_laclip_auto/visualizations/`
