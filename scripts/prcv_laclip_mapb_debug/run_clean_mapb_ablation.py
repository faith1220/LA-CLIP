#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON = "/home/huoyongzhen/.conda/envs/multiads/bin/python"
MAIN_PY = ROOT_DIR / "main.py"
ALLOWED_GPUS = [2, 3, 4, 6]
BUSY_MEMORY_MB = 2000
REFRESH_SECONDS = 30
LOG_ROOT = ROOT_DIR / "logs" / "prcv_laclip_mapb_debug"
REPORT_ROOT = ROOT_DIR / "reports" / "prcv_laclip_mapb_debug"
DIAG_ROOT = REPORT_ROOT / "diagnostics"
RERUN_ROOT = REPORT_ROOT / "rerun"
STATE_JSON = RERUN_ROOT / "runner_state.json"
SUMMARY_CSV = REPORT_ROOT / "MAPB_CLEAN_ABLATION.csv"
SUMMARY_MD = REPORT_ROOT / "MAPB_CLEAN_ABLATION.md"

BASE_ARGS = {
    "data_dir": "./data_local",
    "clip_download_dir": "./download/clip",
    "model": "ViT-L/14@336px",
    "img_size": 518,
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


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_branch_num() -> int:
    return max(len(BASE_ARGS["feature_layers"]), 1) * 3


def flatten_cli(args_dict: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
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


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def tail_text(path: Path, count: int = 100) -> str:
    return "\n".join(read_text(path).splitlines()[-count:])


def sanitize(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def parse_metrics(text: str) -> Dict[str, Optional[float]]:
    patterns = {
        "i_auroc": r"Sample_CLS_AUROC:\s*([0-9.]+)",
        "i_ap": r"Sample_CLS_AP:\s*([0-9.]+)",
        "i_f1": r"Sample_CLS_max-F1:\s*([0-9.]+)",
        "p_auroc": r"Pixel_AUROC:\s*([0-9.]+)",
        "p_ap": r"Pixel_AP:\s*([0-9.]+)",
        "pro": r"Pixel_PRO:\s*([0-9.]+)",
    }
    metrics = {key: None for key in patterns}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def metrics_complete(metrics: Dict[str, Optional[float]]) -> bool:
    return metrics.get("i_auroc") is not None and metrics.get("p_auroc") is not None


def metrics_from_report(report_dir: Path, log_path: Path) -> Dict[str, Optional[float]]:
    best = {key: None for key in ["i_auroc", "i_ap", "i_f1", "p_auroc", "p_ap", "pro"]}
    texts = []
    if report_dir.exists():
        texts.extend(read_text(path) for path in sorted(report_dir.glob("*.txt")))
    if log_path.exists():
        texts.append(read_text(log_path))
    for text in texts:
        parsed = parse_metrics(text)
        if metrics_complete(parsed):
            best = parsed
    return best


def detect_error(text: str, return_code: Optional[int]) -> str:
    checks = [
        ("traceback", r"Traceback"),
        ("oom", r"CUDA out of memory"),
        ("file_not_found", r"FileNotFoundError"),
        ("runtime_error", r"RuntimeError"),
        ("value_error", r"ValueError"),
        ("nan", r"(^|[^A-Za-z])NaN([^A-Za-z]|$)|(^|[^A-Za-z])nan([^A-Za-z]|$)"),
        ("inf", r"(^|[^A-Za-z])inf([^A-Za-z]|$)"),
    ]
    for name, pattern in checks:
        if re.search(pattern, text, flags=re.MULTILINE):
            return name
    if return_code not in (None, 0):
        return f"return_code_{return_code}"
    return ""


def query_gpu_memory() -> Dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_memory: Dict[int, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, mem_str = [part.strip() for part in line.split(",", 1)]
        gpu_memory[int(idx_str)] = int(mem_str)
    return gpu_memory


@dataclass
class Experiment:
    name: str
    phase: str
    requested_branch_num: str
    effective_branch_num: int
    aggregation: str
    epochs: int
    batch_size: int
    report_dir: str
    log_path: str
    debug_json_path: str
    status: str = "pending"
    gpu: Optional[int] = None
    pid: Optional[int] = None
    start_time: str = ""
    end_time: str = ""
    duration_min: Optional[float] = None
    return_code: Optional[int] = None
    error_type: str = ""
    error_tail: str = ""
    metrics: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "i_auroc": None,
            "i_ap": None,
            "i_f1": None,
            "p_auroc": None,
            "p_ap": None,
            "pro": None,
        }
    )

    @property
    def report_path(self) -> Path:
        return ROOT_DIR / self.report_dir

    @property
    def log_file(self) -> Path:
        return ROOT_DIR / self.log_path

    @property
    def debug_json(self) -> Path:
        return ROOT_DIR / self.debug_json_path

    def command(self) -> List[str]:
        args = dict(BASE_ARGS)
        args.update(
            {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "log_dir": self.report_dir,
                "mapb_branch_num": self.effective_branch_num,
                "mapb_aggregation": self.aggregation,
                "mapb_debug_json": self.debug_json_path,
            }
        )
        return [PYTHON, str(MAIN_PY)] + flatten_cli(args)


class Runner:
    def __init__(self, phases: List[str]):
        self.phases = phases
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        DIAG_ROOT.mkdir(parents=True, exist_ok=True)
        RERUN_ROOT.mkdir(parents=True, exist_ok=True)
        self.running: Dict[str, subprocess.Popen] = {}
        self.experiments: List[Experiment] = []
        for phase in phases:
            self.experiments.extend(self.build_phase_experiments(phase))
        self.persist()

    def build_phase_experiments(self, phase: str) -> List[Experiment]:
        default_num = default_branch_num()
        experiments: List[Experiment] = []
        if phase == "smoke":
            branch_settings = [("default", default_num), ("1", 1), ("2", 2), ("4", 4), ("6", 6), ("8", 8)]
            for requested, effective in branch_settings:
                name = f"smoke_branch_{requested}"
                experiments.append(
                    Experiment(
                        name=name,
                        phase=phase,
                        requested_branch_num=requested,
                        effective_branch_num=effective,
                        aggregation="mean",
                        epochs=1,
                        batch_size=2,
                        report_dir=f"reports/prcv_laclip_mapb_debug/{name}",
                        log_path=f"logs/prcv_laclip_mapb_debug/{name}.log",
                        debug_json_path=f"reports/prcv_laclip_mapb_debug/{name}/{name}_mapb_debug.json",
                    )
                )
        elif phase == "rerun":
            branch_settings = [1, 2, 4, 6, 8, default_num]
            for effective in branch_settings:
                name = f"mapb_clean_branch_{effective}" if effective != default_num else "mapb_clean_default_explicit"
                experiments.append(
                    Experiment(
                        name=name,
                        phase=phase,
                        requested_branch_num=str(effective),
                        effective_branch_num=effective,
                        aggregation="mean",
                        epochs=2,
                        batch_size=8,
                        report_dir=f"reports/prcv_laclip_mapb_debug/rerun/{name}",
                        log_path=f"logs/prcv_laclip_mapb_debug/{name}.log",
                        debug_json_path=f"reports/prcv_laclip_mapb_debug/rerun/{name}/{name}_mapb_debug.json",
                    )
                )
        elif phase == "agg":
            branch_num = self.best_or_default_branch_num()
            for aggregation in ["mean", "max", "logsumexp"]:
                name = f"mapb_clean_agg_{aggregation}"
                experiments.append(
                    Experiment(
                        name=name,
                        phase=phase,
                        requested_branch_num=str(branch_num),
                        effective_branch_num=branch_num,
                        aggregation=aggregation,
                        epochs=2,
                        batch_size=8,
                        report_dir=f"reports/prcv_laclip_mapb_debug/rerun/{name}",
                        log_path=f"logs/prcv_laclip_mapb_debug/{name}.log",
                        debug_json_path=f"reports/prcv_laclip_mapb_debug/rerun/{name}/{name}_mapb_debug.json",
                    )
                )
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        return experiments

    def best_or_default_branch_num(self) -> int:
        default_num = default_branch_num()
        candidates = []
        for exp in self.experiments:
            if exp.phase != "rerun":
                continue
            if exp.status != "completed":
                continue
            score = exp.metrics.get("i_auroc")
            if score is not None:
                candidates.append((score, exp.effective_branch_num))
        if not candidates:
            return default_num
        candidates.sort(reverse=True)
        return candidates[0][1]

    def detect_completed(self, exp: Experiment):
        metrics = metrics_from_report(exp.report_path, exp.log_file)
        if metrics_complete(metrics):
            exp.metrics = metrics
            exp.status = "completed"

    def refresh_running(self):
        for name, proc in list(self.running.items()):
            exp = next(item for item in self.experiments if item.name == name)
            rc = proc.poll()
            if rc is None:
                exp.status = "running"
                continue
            exp.return_code = rc
            exp.end_time = now_str()
            if exp.start_time:
                start_dt = datetime.strptime(exp.start_time, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(exp.end_time, "%Y-%m-%d %H:%M:%S")
                exp.duration_min = round((end_dt - start_dt).total_seconds() / 60.0, 2)
            tail = tail_text(exp.log_file, 100)
            exp.error_tail = sanitize(tail)
            exp.error_type = detect_error(tail, rc)
            exp.metrics = metrics_from_report(exp.report_path, exp.log_file)
            if rc == 0 and metrics_complete(exp.metrics) and not exp.error_type:
                exp.status = "completed"
            elif rc == 0 and not exp.error_type:
                exp.status = "suspicious"
            else:
                exp.status = "failed"
            del self.running[name]

    def current_gpu_assignments(self) -> Dict[int, str]:
        assignments = {}
        for exp in self.experiments:
            if exp.status == "running" and exp.gpu is not None:
                assignments[exp.gpu] = exp.name
        return assignments

    def find_free_gpu(self) -> Optional[int]:
        gpu_memory = query_gpu_memory()
        assignments = self.current_gpu_assignments()
        for gpu in ALLOWED_GPUS:
            if gpu in assignments:
                continue
            if gpu_memory.get(gpu, 0) > BUSY_MEMORY_MB:
                continue
            return gpu
        return None

    def launch(self, exp: Experiment, gpu: int):
        exp.report_path.mkdir(parents=True, exist_ok=True)
        exp.log_file.parent.mkdir(parents=True, exist_ok=True)
        command = exp.command()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        with exp.log_file.open("w", encoding="utf-8") as handle:
            handle.write(f"[runner] started_at={now_str()} gpu={gpu}\n")
            handle.write(f"[runner] command={shlex.join(command)}\n\n")
            handle.flush()
            proc = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        exp.status = "running"
        exp.gpu = gpu
        exp.pid = proc.pid
        exp.start_time = now_str()
        self.running[exp.name] = proc

    def pending_for_phase(self, phase: str) -> List[Experiment]:
        return [exp for exp in self.experiments if exp.phase == phase and exp.status not in {"completed", "failed", "suspicious"}]

    def run_phase(self, phase: str):
        for exp in self.experiments:
            if exp.phase != phase:
                continue
            self.detect_completed(exp)
        while True:
            self.refresh_running()
            self.persist()
            phase_items = [exp for exp in self.experiments if exp.phase == phase]
            if all(exp.status in {"completed", "failed", "suspicious"} for exp in phase_items):
                break
            launched = False
            for exp in phase_items:
                if exp.status in {"completed", "failed", "suspicious", "running"}:
                    continue
                self.detect_completed(exp)
                if exp.status == "completed":
                    continue
                gpu = self.find_free_gpu()
                if gpu is None:
                    break
                self.launch(exp, gpu)
                launched = True
            self.persist()
            if not launched:
                time.sleep(REFRESH_SECONDS)

    def persist(self):
        payload = {
            "updated_at": now_str(),
            "allowed_gpus": ALLOWED_GPUS,
            "gpu_memory_mb": query_gpu_memory(),
            "experiments": [asdict(exp) for exp in self.experiments],
        }
        STATE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.write_summary()

    def write_summary(self):
        rows = []
        for exp in self.experiments:
            row = {
                "experiment_name": exp.name,
                "requested_branch_num": exp.requested_branch_num,
                "effective_branch_num": exp.effective_branch_num,
                "abnormal_prompt_count": 1,
                "aggregation": exp.aggregation,
                "score_mode": "prototype",
                "prototype_k": BASE_ARGS["prototype_k"],
                "prototype_fusion_alpha": BASE_ARGS["prototype_fusion_alpha"],
                "seed": BASE_ARGS["seed"],
                "i_auroc": exp.metrics.get("i_auroc"),
                "i_ap": exp.metrics.get("i_ap"),
                "i_f1": exp.metrics.get("i_f1"),
                "p_auroc": exp.metrics.get("p_auroc"),
                "p_ap": exp.metrics.get("p_ap"),
                "pro": exp.metrics.get("pro"),
                "status": exp.status,
                "log_path": exp.log_path,
                "report_path": exp.report_dir,
                "debug_json_path": exp.debug_json_path,
            }
            rows.append(row)
        fieldnames = list(rows[0].keys()) if rows else []
        if rows:
            with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            md_lines = [
                "# MAPB Clean Ablation",
                "",
                "| Experiment | Req Branch | Eff Branch | Aggregation | Status | I-AUROC | I-AP | P-AUROC | P-AP | PRO |",
                "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in rows:
                md_lines.append(
                    f"| {row['experiment_name']} | {row['requested_branch_num']} | {row['effective_branch_num']} | {row['aggregation']} | {row['status']} | "
                    f"{row['i_auroc'] if row['i_auroc'] is not None else ''} | {row['i_ap'] if row['i_ap'] is not None else ''} | "
                    f"{row['p_auroc'] if row['p_auroc'] is not None else ''} | {row['p_ap'] if row['p_ap'] is not None else ''} | "
                    f"{row['pro'] if row['pro'] is not None else ''} |"
                )
            SUMMARY_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run clean MAPB branch and aggregation ablations on GPUs 2/3/4/6")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["smoke", "rerun", "agg"],
        choices=["smoke", "rerun", "agg"],
        help="Which phases to run. They execute in the given order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = Runner(args.phases)
    for phase in args.phases:
        if phase == "agg":
            runner.experiments = [exp for exp in runner.experiments if exp.phase != "agg"]
            runner.experiments.extend(runner.build_phase_experiments("agg"))
        runner.run_phase(phase)
    runner.persist()


if __name__ == "__main__":
    main()
