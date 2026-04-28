#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON = "/home/huoyongzhen/.conda/envs/multiads/bin/python"
MAIN_PY = ROOT_DIR / "main.py"
LOG_ROOT = ROOT_DIR / "logs" / "prcv_laclip_mapb_debug"
REPORT_ROOT = ROOT_DIR / "reports" / "prcv_laclip_mapb_debug" / "rerun"
ALLOWED_GPUS = [2, 3, 4, 6]
BUSY_MEMORY_MB = 2000

BASE_ARGS = {
    "data_dir": "./data_local",
    "clip_download_dir": "./download/clip",
    "model": "ViT-L/14@336px",
    "img_size": 518,
    "batch_size": 8,
    "seed": 122,
    "feature_layers": [6, 12, 18, 24],
    "memory_layers": [6, 12, 18, 24],
    "use_lsar": 1,
    "lsar_bottleneck_ratio": 4,
    "use_mvti": 0,
    "use_mapb": 1,
    "score_mode": "prototype",
    "lambda_proto": 0.1,
    "prototype_k": 4,
    "prototype_momentum": 0.95,
    "prototype_temperature": 0.07,
    "prototype_max_samples": 4096,
    "prototype_fusion_alpha": 0.25,
    "dataset": "mvtec",
    "test_dataset": ["visa"],
    "debug_mapb": 1,
}


def query_gpu_memory() -> Dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_memory = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, mem_str = [part.strip() for part in line.split(",", 1)]
        gpu_memory[int(idx_str)] = int(mem_str)
    return gpu_memory


def flatten_cli(args_dict):
    tokens = []
    for key, value in args_dict.items():
        if value is None:
            continue
        cli_key = f"--{key}"
        if isinstance(value, (list, tuple)):
            tokens.append(cli_key)
            tokens.extend([str(item) for item in value])
        else:
            tokens.extend([cli_key, str(value)])
    return tokens


def parse_metrics(text: str):
    patterns = {
        "i_auroc": r"Sample_CLS_AUROC:\s*([0-9.]+)",
        "i_ap": r"Sample_CLS_AP:\s*([0-9.]+)",
        "p_auroc": r"Pixel_AUROC:\s*([0-9.]+)",
        "p_ap": r"Pixel_AP:\s*([0-9.]+)",
        "pro": r"Pixel_PRO:\s*([0-9.]+)",
    }
    values = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        values[key] = matches[-1] if matches else ""
    return values


def find_free_gpu() -> Optional[int]:
    gpu_memory = query_gpu_memory()
    for gpu in ALLOWED_GPUS:
        if gpu_memory.get(gpu, 0) <= BUSY_MEMORY_MB:
            return gpu
    return None


def run_eval(weight_dir: str, branch_num: int, aggregation: str):
    name = f"mapb_clean_agg_{aggregation}"
    log_path = LOG_ROOT / f"{name}.log"
    report_dir = REPORT_ROOT / name
    debug_json = report_dir / f"{name}_mapb_debug.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    args = dict(BASE_ARGS)
    args.update(
        {
            "log_dir": str(report_dir.relative_to(ROOT_DIR)),
            "mapb_branch_num": branch_num,
            "mapb_aggregation": aggregation,
            "mapb_debug_json": str(debug_json.relative_to(ROOT_DIR)),
            "weight": weight_dir,
        }
    )
    command = [PYTHON, str(MAIN_PY)] + flatten_cli(args)
    while True:
        gpu = find_free_gpu()
        if gpu is not None:
            break
        time.sleep(30)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[agg-eval] gpu={gpu}\n")
        handle.write(f"[agg-eval] command={shlex.join(command)}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Run MAPB aggregation ablation in eval-only mode from a clean checkpoint")
    parser.add_argument("--weight_dir", required=True, help="Checkpoint directory to load with --weight")
    parser.add_argument("--branch_num", type=int, required=True, help="Effective branch count that matches the checkpoint")
    parser.add_argument("--aggregations", nargs="+", default=["mean", "max", "logsumexp"])
    args = parser.parse_args()

    for aggregation in args.aggregations:
        run_eval(args.weight_dir, args.branch_num, aggregation)


if __name__ == "__main__":
    main()
