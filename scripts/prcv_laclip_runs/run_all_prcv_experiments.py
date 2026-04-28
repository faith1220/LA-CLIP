#!/usr/bin/env python3
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
PYTHON = "/home/huoyongzhen/.conda/envs/multiads/bin/python"
MAIN_PY = ROOT_DIR / "main.py"
AUTO_LOG_ROOT = ROOT_DIR / "logs" / "prcv_laclip_auto"
AUTO_REPORT_ROOT = ROOT_DIR / "reports" / "prcv_laclip_auto"
COMMAND_ROOT = AUTO_REPORT_ROOT / "commands"
STATE_JSON = AUTO_REPORT_ROOT / "scheduler_state.json"
STATE_MD = AUTO_REPORT_ROOT / "scheduler_state.md"
SUMMARY_CSV = AUTO_REPORT_ROOT / "experiment_summary.csv"
SUMMARY_MD = AUTO_REPORT_ROOT / "experiment_summary.md"
VIS_STATUS_JSON = AUTO_REPORT_ROOT / "visualizations" / "visualization_status.json"
LEGACY_LOG_ROOT = ROOT_DIR / "logs" / "prcv_laclip"
LEGACY_REPORT_ROOT = ROOT_DIR / "reports" / "prcv_laclip"
MAIN_HELP_PATH = LEGACY_REPORT_ROOT / "main_help.txt"

ALLOWED_GPUS = [2, 3, 4, 6]
FORBIDDEN_GPUS = [0, 1, 5, 7]
BUSY_MEMORY_MB = 2000
REFRESH_SECONDS = 60

COMPLETE_STATUSES = {"completed", "completed_external", "completed_reused"}
FINAL_STATUSES = COMPLETE_STATUSES | {"failed", "unsupported", "skipped", "suspicious"}

BASE_ARGS = {
    "data_dir": "./data_local",
    "clip_download_dir": "./download/clip",
    "model": "ViT-L/14@336px",
    "img_size": 518,
    "batch_size": 8,
    "epochs": 2,
    "seed": 122,
    "feature_layers": [6, 12, 18, 24],
    "memory_layers": [6, 12, 18, 24],
    "score_mode": "prototype",
    "lambda_proto": 0.1,
    "prototype_k": 4,
    "prototype_momentum": 0.95,
    "prototype_temperature": 0.07,
    "prototype_max_samples": 4096,
    "prototype_fusion_alpha": 0.25,
    "use_lsar": 1,
    "lsar_bottleneck_ratio": 4,
    "lsar_zero_init": 1,
    "use_mvti": 1,
    "mvti_views": 2,
    "use_mapb": 1,
    "mapb_branch_num": 0,
    "mapb_aggregation": "mean",
}

DATASET_PATH_CHECKS = {
    "mvtec": ROOT_DIR / "data_local" / "mvtec",
    "visa": ROOT_DIR / "data_local" / "visa",
    "clinic": ROOT_DIR / "data_local" / "CVC-ClinicDB",
    "colon": ROOT_DIR / "data_local" / "CVC-ColonDB",
    "kvasir": ROOT_DIR / "data_local" / "Kvasir",
}

GROUP_ORDER = [
    "Main Results",
    "Component Ablation",
    "MAPB Ablation",
    "LSAR / LSVA Ablation",
    "MVTI Ablation",
    "Layer Selection",
    "Prototype / Score Analysis",
    "Visualization Export Status",
    "Failed / Unsupported Experiments",
]


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def tail_text(path: Path, line_count: int = 100) -> str:
    lines = read_text(path).splitlines()
    return "\n".join(lines[-line_count:])


def sanitize_error_tail(text: str, limit: int = 3000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


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


def infer_effective_mapb_branch_num(args_dict: Dict[str, object]) -> int:
    requested = int(args_dict.get("mapb_branch_num", args_dict.get("mapb_branch_count", 0)) or 0)
    feature_layers = args_dict.get("feature_layers") or []
    default_branch_num = max(len(feature_layers), 1) * 3
    if requested <= 0:
        return default_branch_num
    return min(requested, default_branch_num)


def parse_metrics_from_text(text: str) -> Dict[str, Optional[float]]:
    patterns = {
        "i_auroc": r"Sample_CLS_AUROC:\s*([0-9.]+)",
        "i_ap": r"Sample_CLS_AP:\s*([0-9.]+)",
        "i_f1": r"Sample_CLS_max-F1:\s*([0-9.]+)",
        "p_auroc": r"Pixel_AUROC:\s*([0-9.]+)",
        "p_ap": r"Pixel_AP:\s*([0-9.]+)",
        "pro": r"Pixel_PRO:\s*([0-9.]+)",
    }
    metrics: Dict[str, Optional[float]] = {key: None for key in patterns}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def metrics_are_complete(metrics: Dict[str, Optional[float]]) -> bool:
    image_ready = metrics.get("i_auroc") is not None or metrics.get("i_ap") is not None
    pixel_ready = metrics.get("p_auroc") is not None or metrics.get("p_ap") is not None or metrics.get("pro") is not None
    return image_ready or pixel_ready


def detect_error_type(text: str, return_code: Optional[int]) -> str:
    patterns = [
        ("traceback", r"Traceback"),
        ("oom", r"CUDA out of memory"),
        ("file_not_found", r"FileNotFoundError"),
        ("runtime_error", r"RuntimeError"),
        ("value_error", r"ValueError"),
        ("nan", r"(^|[^A-Za-z])NaN([^A-Za-z]|$)|(^|[^A-Za-z])nan([^A-Za-z]|$)"),
        ("inf", r"(^|[^A-Za-z])inf([^A-Za-z]|$)"),
    ]
    for error_name, pattern in patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            return error_name
    if return_code not in (None, 0):
        return f"return_code_{return_code}"
    return ""


def find_metrics_in_dir(report_dir: Path, log_path: Optional[Path] = None) -> Dict[str, Optional[float]]:
    text_candidates: List[str] = []
    if report_dir.exists():
        for txt_file in sorted(report_dir.glob("*.txt")):
            text_candidates.append(read_text(txt_file))
    if log_path is not None and log_path.exists():
        text_candidates.append(read_text(log_path))
    best_metrics: Dict[str, Optional[float]] = {
        "i_auroc": None,
        "i_ap": None,
        "i_f1": None,
        "p_auroc": None,
        "p_ap": None,
        "pro": None,
    }
    for text in text_candidates:
        metrics = parse_metrics_from_text(text)
        if metrics_are_complete(metrics):
            best_metrics = metrics
    return best_metrics


def visualization_completed() -> bool:
    if not VIS_STATUS_JSON.exists():
        return False
    try:
        payload = json.loads(VIS_STATUS_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("cross_domain_heatmap_comparison") == "completed"
        and payload.get("mvti_stabilization") == "completed"
    )


def query_gpu_memory() -> Dict[int, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    gpu_memory: Dict[int, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, mem_str = [part.strip() for part in line.split(",", 1)]
        gpu_memory[int(idx_str)] = int(mem_str)
    return gpu_memory


def find_process_for_report_dir(report_dir: Path) -> Optional[int]:
    result = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=False)
    target = str(report_dir)
    for line in result.stdout.splitlines():
        if target not in line:
            continue
        if "python main.py" in line or "bash -lc" in line:
            pid_str = line.strip().split(None, 1)[0]
            try:
                return int(pid_str)
            except ValueError:
                return None
    return None


def tmux_session_exists(session_name: str) -> bool:
    if not session_name:
        return False
    result = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True, text=True, check=False)
    return result.returncode == 0


def dataset_is_available(dataset_name: str) -> bool:
    dataset_path = DATASET_PATH_CHECKS.get(dataset_name)
    return dataset_path is not None and dataset_path.exists()


@dataclass
class Experiment:
    name: str
    group: str
    source_dataset: str
    target_dataset: str
    arg_overrides: Dict[str, object] = field(default_factory=dict)
    kind: str = "run"
    weight_from: Optional[str] = None
    alias_of: Optional[str] = None
    unsupported_reason: str = ""
    external_report_dir: str = ""
    external_log_path: str = ""
    external_tmux_session: str = ""
    visualization_script: str = ""
    note: str = ""

    status: str = "pending"
    gpu: Optional[int] = None
    pid: Optional[int] = None
    start_time: str = ""
    end_time: str = ""
    duration_min: Optional[float] = None
    return_code: Optional[int] = None
    error_type: str = ""
    error_tail: str = ""
    source_ref: str = "auto"
    metrics: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "i_auroc": None,
        "i_ap": None,
        "i_f1": None,
        "p_auroc": None,
        "p_ap": None,
        "pro": None,
    })

    @property
    def report_dir(self) -> Path:
        return AUTO_REPORT_ROOT / self.name

    @property
    def log_path(self) -> Path:
        return AUTO_LOG_ROOT / f"{self.name}.log"

    @property
    def command_path(self) -> Path:
        return COMMAND_ROOT / f"{self.name}.sh"

    @property
    def external_report_path(self) -> Optional[Path]:
        return Path(self.external_report_dir) if self.external_report_dir else None

    @property
    def external_log(self) -> Optional[Path]:
        return Path(self.external_log_path) if self.external_log_path else None

    def checkpoint_dir(self) -> Path:
        return self.report_dir / "checkpoints"

    def external_checkpoint_dir(self) -> Optional[Path]:
        if self.external_report_path is None:
            return None
        return self.external_report_path / "checkpoints"


class Scheduler:
    def __init__(self):
        AUTO_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        COMMAND_ROOT.mkdir(parents=True, exist_ok=True)
        AUTO_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.running: Dict[str, subprocess.Popen] = {}
        self.experiments: Dict[str, Experiment] = self._build_experiments()
        self._install_signal_handlers()
        self._write_main_help()
        self._write_command_stubs()
        self.refresh_states(initial=True)
        self.persist_state()

    def _install_signal_handlers(self):
        def _handle_signal(signum, _frame):
            self.stop_requested = True
            for experiment_name, process in list(self.running.items()):
                if process.poll() is None:
                    process.terminate()
                    exp = self.experiments[experiment_name]
                    exp.status = "failed"
                    exp.error_type = f"terminated_by_signal_{signum}"
                    exp.error_tail = sanitize_error_tail(tail_text(exp.log_path, 100))
            self.persist_state()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def _write_main_help(self):
        MAIN_HELP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MAIN_HELP_PATH.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [PYTHON, str(MAIN_PY), "--help"],
                cwd=ROOT_DIR,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def _build_experiments(self) -> Dict[str, Experiment]:
        experiments: List[Experiment] = []

        def exp(name, group, source, target, **kwargs):
            experiments.append(Experiment(name=name, group=group, source_dataset=source, target_dataset=target, **kwargs))

        exp(
            "main_m2v_full",
            "Main Results",
            "mvtec",
            "visa",
            external_report_dir=str(LEGACY_REPORT_ROOT / "mvtec_to_visa_full"),
            external_log_path=str(LEGACY_LOG_ROOT / "mvtec_to_visa_full.log"),
            external_tmux_session="laclip_main_m2v",
        )
        exp(
            "main_v2m_full",
            "Main Results",
            "visa",
            "mvtec",
            external_report_dir=str(LEGACY_REPORT_ROOT / "visa_to_mvtec_full"),
            external_log_path=str(LEGACY_LOG_ROOT / "visa_to_mvtec_full.log"),
            external_tmux_session="laclip_main_v2m",
        )
        exp("main_m2clinicdb", "Main Results", "mvtec", "clinic", kind="run", weight_from="main_m2v_full")
        exp("main_m2colondb", "Main Results", "mvtec", "colon", kind="run", weight_from="main_m2v_full")
        exp("main_m2kvasir", "Main Results", "mvtec", "kvasir", kind="run", weight_from="main_m2v_full")

        exp(
            "comp_baseline",
            "Component Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 0, "use_lsar": 0, "use_mvti": 0, "score_mode": "clip"},
        )
        exp(
            "comp_mapb_only",
            "Component Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 0, "use_mvti": 0, "score_mode": "prototype"},
        )
        exp(
            "comp_lsar_only",
            "Component Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 0, "use_lsar": 1, "use_mvti": 0, "score_mode": "clip"},
        )
        exp(
            "comp_mapb_lsar",
            "Component Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype"},
            kind="run",
            weight_from="main_m2v_full",
        )
        exp("comp_full", "Component Ablation", "mvtec", "visa", kind="alias", alias_of="main_m2v_full")
        exp(
            "comp_v2m_baseline",
            "Component Ablation",
            "visa",
            "mvtec",
            arg_overrides={"use_mapb": 0, "use_lsar": 0, "use_mvti": 0, "score_mode": "clip"},
        )
        exp(
            "comp_v2m_mapb_lsar",
            "Component Ablation",
            "visa",
            "mvtec",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype"},
            kind="run",
            weight_from="main_v2m_full",
        )
        exp("comp_v2m_full", "Component Ablation", "visa", "mvtec", kind="alias", alias_of="main_v2m_full")

        for branch_count in [1, 2, 4, 6]:
            exp(
                f"mapb_branch_{branch_count}",
                "MAPB Ablation",
                "mvtec",
                "visa",
                arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "mapb_branch_num": branch_count},
            )
        exp("mapb_agg_mean", "MAPB Ablation", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")
        exp(
            "mapb_agg_max",
            "MAPB Ablation",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "mapb_aggregation": "max"},
        )
        exp(
            "mapb_agg_logsumexp",
            "MAPB Ablation",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "mapb_aggregation": "logsumexp"},
        )

        exp("lsva_shared_only", "LSAR / LSVA Ablation", "mvtec", "visa", kind="alias", alias_of="comp_mapb_only")
        exp(
            "lsva_lsar_ratio_2",
            "LSAR / LSVA Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "lsar_bottleneck_ratio": 2},
        )
        exp("lsva_lsar_ratio_4", "LSAR / LSVA Ablation", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")
        exp(
            "lsva_lsar_ratio_8",
            "LSAR / LSVA Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "lsar_bottleneck_ratio": 8},
        )
        exp(
            "lsva_no_zero_init",
            "LSAR / LSVA Ablation",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "lsar_zero_init": 0},
        )
        exp(
            "lsva_independent_adaptor",
            "LSAR / LSVA Ablation",
            "mvtec",
            "visa",
            kind="unsupported",
            unsupported_reason="Current repository has no per-layer independent adaptor CLI or module split; adding it would change model structure beyond a minimal scheduler patch.",
        )

        exp("mvti_view_1", "MVTI Ablation", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")
        exp("mvti_view_2", "MVTI Ablation", "mvtec", "visa", kind="alias", alias_of="main_m2v_full")
        exp(
            "mvti_view_4",
            "MVTI Ablation",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 1, "mvti_views": 4, "score_mode": "prototype"},
        )
        exp(
            "mvti_view_8",
            "MVTI Ablation",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 1, "mvti_views": 8, "score_mode": "prototype"},
        )

        exp(
            "layer_last_only",
            "Layer Selection",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "feature_layers": [24], "memory_layers": [24]},
        )
        exp(
            "layer_shallow",
            "Layer Selection",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "feature_layers": [6, 12], "memory_layers": [6, 12]},
        )
        exp(
            "layer_deep",
            "Layer Selection",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "feature_layers": [18, 24], "memory_layers": [18, 24]},
        )
        exp("layer_multilayer_default", "Layer Selection", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")

        exp(
            "proto_off_or_default_score",
            "Prototype / Score Analysis",
            "mvtec",
            "visa",
            kind="unsupported",
            unsupported_reason="Current clean code couples MAPB enablement and prototype scoring; disabling prototype score while keeping MAPB active would require a semantic change to the scoring path.",
        )
        exp(
            "proto_k_2",
            "Prototype / Score Analysis",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "prototype_k": 2},
        )
        exp("proto_k_4", "Prototype / Score Analysis", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")
        exp(
            "proto_k_8",
            "Prototype / Score Analysis",
            "mvtec",
            "visa",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "prototype_k": 8},
        )
        exp(
            "proto_alpha_0",
            "Prototype / Score Analysis",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "prototype_fusion_alpha": 0.0},
        )
        exp("proto_alpha_025", "Prototype / Score Analysis", "mvtec", "visa", kind="alias", alias_of="comp_mapb_lsar")
        exp(
            "proto_alpha_05",
            "Prototype / Score Analysis",
            "mvtec",
            "visa",
            kind="run",
            weight_from="main_m2v_full",
            arg_overrides={"use_mapb": 1, "use_lsar": 1, "use_mvti": 0, "score_mode": "prototype", "prototype_fusion_alpha": 0.5},
        )

        exp(
            "visualization_export",
            "Visualization Export Status",
            "mvtec",
            "visa",
            kind="visualization",
            visualization_script=str(ROOT_DIR / "scripts" / "prcv_laclip_runs" / "export_visualizations.py"),
            note="Exports final heatmap comparison and MVTI stabilization with available checkpoints; branch/layer internals are reported as partial or unsupported.",
        )

        return {experiment.name: experiment for experiment in experiments}

    def _write_command_stubs(self):
        for experiment in self.experiments.values():
            experiment.command_path.parent.mkdir(parents=True, exist_ok=True)
            if experiment.kind == "unsupported":
                experiment.command_path.write_text(
                    f"#!/usr/bin/env bash\n# unsupported: {experiment.unsupported_reason}\n",
                    encoding="utf-8",
                )

    def get_effective_args(self, experiment: Experiment) -> Dict[str, object]:
        args = deepcopy(BASE_ARGS)
        args.update(experiment.arg_overrides)
        args["dataset"] = experiment.source_dataset
        args["test_dataset"] = [experiment.target_dataset]
        args["log_dir"] = str(experiment.report_dir)
        return args

    def resolve_checkpoint_dir(self, experiment_name: str) -> Optional[Path]:
        source_experiment = self.experiments[experiment_name]
        if source_experiment.status in COMPLETE_STATUSES:
            if source_experiment.source_ref == "external":
                checkpoint_dir = source_experiment.external_checkpoint_dir()
                if checkpoint_dir is not None and checkpoint_dir.exists():
                    return checkpoint_dir
            checkpoint_dir = source_experiment.checkpoint_dir()
            if checkpoint_dir.exists():
                return checkpoint_dir
        return None

    def build_command(self, experiment: Experiment, weight_dir: Optional[Path] = None) -> List[str]:
        if experiment.kind == "visualization":
            methods = {
                "baseline_weight": self.resolve_checkpoint_dir("comp_baseline"),
                "mapb_lsar_weight": self.resolve_checkpoint_dir("main_m2v_full") or self.resolve_checkpoint_dir("comp_mapb_lsar"),
                "full_weight": self.resolve_checkpoint_dir("main_m2v_full"),
            }
            command = [
                PYTHON,
                experiment.visualization_script,
                "--data_dir",
                "./data_local",
                "--clip_download_dir",
                "./download/clip",
                "--output_dir",
                str(AUTO_REPORT_ROOT / "visualizations"),
                "--source_dataset",
                experiment.source_dataset,
                "--target_dataset",
                experiment.target_dataset,
            ]
            for cli_name, weight_path in methods.items():
                if weight_path is not None:
                    command.extend([f"--{cli_name}", str(weight_path)])
            return command

        args = self.get_effective_args(experiment)
        command = [PYTHON, str(MAIN_PY)]
        command.extend(flatten_cli(args))
        if weight_dir is not None:
            command.extend(["--weight", str(weight_dir)])
        return command

    def write_command_file(self, experiment: Experiment, command: List[str], gpu_id: Optional[int]):
        export_prefix = ""
        if gpu_id is not None:
            export_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 "
        command_text = export_prefix + shlex.join(command)
        experiment.command_path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {shlex.quote(str(ROOT_DIR))}\n"
            f"{command_text}\n",
            encoding="utf-8",
        )

    def can_run(self, experiment: Experiment) -> bool:
        if experiment.kind == "unsupported":
            return False
        if experiment.kind == "alias":
            return False
        if experiment.kind == "visualization":
            return self.numeric_queue_finished()
        if experiment.weight_from:
            return self.resolve_checkpoint_dir(experiment.weight_from) is not None
        return True

    def numeric_queue_finished(self) -> bool:
        for experiment in self.experiments.values():
            if experiment.group == "Visualization Export Status":
                continue
            if experiment.status not in FINAL_STATUSES:
                return False
        return True

    def mark_alias_status(self, experiment: Experiment):
        if not experiment.alias_of:
            return
        source_experiment = self.experiments[experiment.alias_of]
        if source_experiment.status in COMPLETE_STATUSES:
            experiment.status = "completed_reused"
            experiment.source_ref = source_experiment.source_ref or "alias"
            experiment.metrics = deepcopy(source_experiment.metrics)
            experiment.report_dir.mkdir(parents=True, exist_ok=True)
            experiment.command_path.write_text(
                "#!/usr/bin/env bash\n"
                f"# reused from {source_experiment.name}\n",
                encoding="utf-8",
            )
        elif source_experiment.status in {"failed", "suspicious"}:
            experiment.status = "skipped"
            experiment.error_type = f"alias_source_{source_experiment.status}"
        else:
            experiment.status = "waiting"

    def mark_external_status(self, experiment: Experiment):
        if not experiment.external_report_path:
            return
        metrics = find_metrics_in_dir(experiment.external_report_path, experiment.external_log)
        if metrics_are_complete(metrics):
            experiment.status = "completed_external"
            experiment.metrics = metrics
            experiment.source_ref = "external"
            return

        external_pid = find_process_for_report_dir(experiment.external_report_path)
        if external_pid is not None or tmux_session_exists(experiment.external_tmux_session):
            experiment.status = "running_external"
            experiment.pid = external_pid
            experiment.source_ref = "external"
            return

    def refresh_states(self, initial: bool = False):
        for experiment_name, process in list(self.running.items()):
            experiment = self.experiments[experiment_name]
            return_code = process.poll()
            if return_code is None:
                experiment.status = "running"
                continue
            experiment.return_code = return_code
            experiment.end_time = now_str()
            if experiment.start_time:
                start_dt = datetime.strptime(experiment.start_time, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(experiment.end_time, "%Y-%m-%d %H:%M:%S")
                experiment.duration_min = round((end_dt - start_dt).total_seconds() / 60.0, 2)
            log_tail = tail_text(experiment.log_path, 100)
            error_type = detect_error_type(log_tail, return_code)
            metrics = find_metrics_in_dir(experiment.report_dir, experiment.log_path)
            experiment.metrics = metrics
            experiment.error_tail = sanitize_error_tail(log_tail)
            experiment.error_type = error_type
            if return_code == 0 and metrics_are_complete(metrics) and not error_type:
                experiment.status = "completed"
            elif return_code == 0 and metrics_are_complete(metrics) and error_type in {"nan", "inf"}:
                experiment.status = "failed"
            elif return_code == 0 and not metrics_are_complete(metrics):
                experiment.status = "suspicious"
            else:
                experiment.status = "failed"
            del self.running[experiment_name]

        for experiment in self.experiments.values():
            if experiment.status in {"running", "completed", "completed_external", "completed_reused", "failed", "unsupported", "suspicious", "skipped"} and not initial:
                continue

            if experiment.kind == "unsupported":
                experiment.status = "unsupported"
                experiment.error_type = "unsupported"
                experiment.error_tail = experiment.unsupported_reason
                continue

            if experiment.kind == "visualization":
                if visualization_completed():
                    experiment.status = "completed"
                    experiment.error_type = ""
                    experiment.error_tail = ""
                elif experiment.status not in FINAL_STATUSES:
                    experiment.status = "pending"
                continue

            if experiment.kind == "alias":
                self.mark_alias_status(experiment)
                continue

            if experiment.external_report_path is not None:
                previous_status = experiment.status
                self.mark_external_status(experiment)
                if experiment.status in {"completed_external", "running_external"}:
                    continue
                if previous_status == "running_external":
                    experiment.status = "pending"

            metrics = find_metrics_in_dir(experiment.report_dir, experiment.log_path)
            if metrics_are_complete(metrics):
                experiment.status = "completed"
                experiment.metrics = metrics
                continue

            auto_pid = find_process_for_report_dir(experiment.report_dir)
            if auto_pid is not None and experiment_name_from_report_dir(experiment.report_dir) == experiment.name and experiment.name not in self.running:
                experiment.status = "running_external"
                experiment.pid = auto_pid
                experiment.source_ref = "auto_detected"
                continue

            if experiment.status not in FINAL_STATUSES:
                experiment.status = "pending"

        self.persist_state()

    def find_free_gpu(self) -> Optional[int]:
        gpu_memory = query_gpu_memory()
        for gpu_id in FORBIDDEN_GPUS:
            if gpu_id in self.running_gpu_assignments():
                raise RuntimeError(f"Scheduler attempted to use forbidden GPU {gpu_id}")
        for gpu_id in ALLOWED_GPUS:
            if gpu_id in self.running_gpu_assignments():
                continue
            if gpu_memory.get(gpu_id, 0) > BUSY_MEMORY_MB:
                continue
            return gpu_id
        return None

    def running_gpu_assignments(self) -> Dict[int, str]:
        assignments = {}
        for experiment_name, process in self.running.items():
            if process.poll() is None:
                gpu_id = self.experiments[experiment_name].gpu
                if gpu_id is not None:
                    assignments[gpu_id] = experiment_name
        return assignments

    def next_runnable_experiment(self) -> Optional[Experiment]:
        for experiment in self.experiments.values():
            if experiment.status in FINAL_STATUSES | {"running", "running_external", "completed_external", "completed_reused"}:
                continue
            if not dataset_is_available(experiment.source_dataset):
                experiment.status = "unsupported"
                experiment.error_type = "missing_source_dataset"
                experiment.error_tail = f"Missing source dataset path for {experiment.source_dataset}: {DATASET_PATH_CHECKS.get(experiment.source_dataset)}"
                continue
            if experiment.target_dataset in DATASET_PATH_CHECKS and not dataset_is_available(experiment.target_dataset):
                experiment.status = "unsupported"
                experiment.error_type = "missing_target_dataset"
                experiment.error_tail = f"Missing target dataset path for {experiment.target_dataset}: {DATASET_PATH_CHECKS.get(experiment.target_dataset)}"
                continue
            if self.can_run(experiment):
                return experiment
            experiment.status = "waiting"
        return None

    def launch_experiment(self, experiment: Experiment, gpu_id: int):
        weight_dir = self.resolve_checkpoint_dir(experiment.weight_from) if experiment.weight_from else None
        command = self.build_command(experiment, weight_dir=weight_dir)
        self.write_command_file(experiment, command, gpu_id)
        experiment.report_dir.mkdir(parents=True, exist_ok=True)
        experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
        with experiment.log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"[scheduler] started_at={now_str()} gpu={gpu_id}\n")
            log_handle.write(f"[scheduler] command={shlex.join(command)}\n\n")
            log_handle.flush()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        experiment.status = "running"
        experiment.gpu = gpu_id
        experiment.pid = process.pid
        experiment.start_time = now_str()
        experiment.end_time = ""
        experiment.return_code = None
        experiment.error_type = ""
        experiment.error_tail = ""
        experiment.source_ref = "auto"
        self.running[experiment.name] = process

    def persist_state(self):
        gpu_memory = query_gpu_memory()
        state_payload = {
            "updated_at": now_str(),
            "allowed_gpus": ALLOWED_GPUS,
            "forbidden_gpus": FORBIDDEN_GPUS,
            "gpu_memory_mb": gpu_memory,
            "running_assignments": self.running_gpu_assignments(),
            "experiments": {
                name: {
                    **asdict(experiment),
                    "report_dir": str(experiment.report_dir),
                    "log_path": str(experiment.log_path),
                    "command_path": str(experiment.command_path),
                    "checkpoint_path": str(experiment.checkpoint_dir()),
                }
                for name, experiment in self.experiments.items()
            },
        }
        STATE_JSON.write_text(json.dumps(state_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.write_summary_csv()
        self.write_summary_md()
        self.write_state_md(gpu_memory)

    def write_summary_csv(self):
        fieldnames = [
            "experiment_name",
            "group",
            "source_dataset",
            "target_dataset",
            "gpu",
            "status",
            "start_time",
            "end_time",
            "duration_min",
            "seed",
            "use_mapb",
            "use_lsar",
            "use_mvti",
            "branch_num",
            "aggregation",
            "lsar_bottleneck_ratio",
            "mvti_views",
            "feature_layers",
            "memory_layers",
            "score_mode",
            "prototype_k",
            "prototype_fusion_alpha",
            "i_auroc",
            "i_ap",
            "i_f1",
            "p_auroc",
            "p_ap",
            "pro",
            "log_path",
            "report_path",
            "command_path",
            "error_type",
            "error_tail",
        ]
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for experiment in self.experiments.values():
                effective_args = self.get_effective_args(experiment) if experiment.kind != "visualization" else {}
                writer.writerow(
                    {
                        "experiment_name": experiment.name,
                        "group": experiment.group,
                        "source_dataset": experiment.source_dataset,
                        "target_dataset": experiment.target_dataset,
                        "gpu": experiment.gpu if experiment.gpu is not None else "",
                        "status": experiment.status,
                        "start_time": experiment.start_time,
                        "end_time": experiment.end_time,
                        "duration_min": experiment.duration_min if experiment.duration_min is not None else "",
                        "seed": effective_args.get("seed", ""),
                        "use_mapb": effective_args.get("use_mapb", ""),
                        "use_lsar": effective_args.get("use_lsar", ""),
                        "use_mvti": effective_args.get("use_mvti", ""),
                        "branch_num": infer_effective_mapb_branch_num(effective_args) if effective_args else "",
                        "aggregation": effective_args.get("mapb_aggregation", ""),
                        "lsar_bottleneck_ratio": effective_args.get("lsar_bottleneck_ratio", ""),
                        "mvti_views": effective_args.get("mvti_views", ""),
                        "feature_layers": " ".join(map(str, effective_args.get("feature_layers", []))) if effective_args.get("feature_layers") else "",
                        "memory_layers": " ".join(map(str, effective_args.get("memory_layers", []))) if effective_args.get("memory_layers") else "",
                        "score_mode": effective_args.get("score_mode", ""),
                        "prototype_k": effective_args.get("prototype_k", ""),
                        "prototype_fusion_alpha": effective_args.get("prototype_fusion_alpha", ""),
                        "i_auroc": experiment.metrics.get("i_auroc", ""),
                        "i_ap": experiment.metrics.get("i_ap", ""),
                        "i_f1": experiment.metrics.get("i_f1", ""),
                        "p_auroc": experiment.metrics.get("p_auroc", ""),
                        "p_ap": experiment.metrics.get("p_ap", ""),
                        "pro": experiment.metrics.get("pro", ""),
                        "log_path": str(experiment.log_path),
                        "report_path": str(experiment.report_dir if experiment.source_ref != "external" else experiment.external_report_path or experiment.report_dir),
                        "command_path": str(experiment.command_path),
                        "error_type": experiment.error_type,
                        "error_tail": experiment.error_tail.replace("\n", " ")[:1000],
                    }
                )

    def write_group_table(self, group_name: str) -> List[str]:
        lines = [f"## {group_name}", ""]
        group_experiments = [exp for exp in self.experiments.values() if exp.group == group_name]
        if not group_experiments:
            lines.extend(["No experiments.", ""])
            return lines
        lines.extend([
            "| Experiment | Source | Target | Status | GPU | I-AUROC | I-AP | P-AUROC | P-AP | PRO | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for experiment in group_experiments:
            note = experiment.note
            if experiment.error_type and experiment.status not in COMPLETE_STATUSES:
                note = f"{note} {experiment.error_type}".strip()
            lines.append(
                "| {name} | {src} | {tgt} | {status} | {gpu} | {i_auroc} | {i_ap} | {p_auroc} | {p_ap} | {pro} | {note} |".format(
                    name=experiment.name,
                    src=experiment.source_dataset,
                    tgt=experiment.target_dataset,
                    status=experiment.status,
                    gpu=experiment.gpu if experiment.gpu is not None else "",
                    i_auroc=self._fmt_metric(experiment.metrics.get("i_auroc")),
                    i_ap=self._fmt_metric(experiment.metrics.get("i_ap")),
                    p_auroc=self._fmt_metric(experiment.metrics.get("p_auroc")),
                    p_ap=self._fmt_metric(experiment.metrics.get("p_ap")),
                    pro=self._fmt_metric(experiment.metrics.get("pro")),
                    note=note,
                )
            )
        lines.append("")
        return lines

    def _fmt_metric(self, value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"

    def write_summary_md(self):
        lines = [
            "# PRCV Experiment Summary",
            "",
            f"- Updated at: {now_str()}",
            f"- Root: {ROOT_DIR}",
            f"- Python: {PYTHON}",
            f"- Allowed GPUs: {ALLOWED_GPUS}",
            "",
        ]
        for group_name in GROUP_ORDER:
            lines.extend(self.write_group_table(group_name))

        failed_or_unsupported = [exp for exp in self.experiments.values() if exp.status in {"failed", "unsupported", "suspicious", "skipped"}]
        lines.extend(["## Failed / Unsupported Details", ""])
        if not failed_or_unsupported:
            lines.append("None.")
            lines.append("")
        else:
            for experiment in failed_or_unsupported:
                lines.append(f"### {experiment.name}")
                lines.append(f"- Status: {experiment.status}")
                if experiment.error_type:
                    lines.append(f"- Error Type: {experiment.error_type}")
                if experiment.error_tail:
                    lines.append("- Error Tail:")
                    lines.append("```text")
                    lines.append(experiment.error_tail)
                    lines.append("```")
                lines.append("")
        SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    def write_state_md(self, gpu_memory: Dict[int, int]):
        running_experiments = [exp for exp in self.experiments.values() if exp.status in {"running", "running_external"}]
        lines = [
            "# Scheduler State",
            "",
            f"- Updated at: {now_str()}",
            f"- Refresh seconds: {REFRESH_SECONDS}",
            f"- Allowed GPUs: {ALLOWED_GPUS}",
            "",
            "## GPU Memory",
            "",
            "| GPU | Memory Used (MiB) | Busy Threshold |",
            "| --- | ---: | ---: |",
        ]
        for gpu_id in sorted(gpu_memory):
            lines.append(f"| {gpu_id} | {gpu_memory[gpu_id]} | {BUSY_MEMORY_MB} |")
        lines.extend(["", "## Running", ""])
        if not running_experiments:
            lines.extend(["None.", ""])
        else:
            lines.extend([
                "| Experiment | Status | GPU | PID | Report Dir | Log |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ])
            for experiment in running_experiments:
                lines.append(
                    f"| {experiment.name} | {experiment.status} | {experiment.gpu or ''} | {experiment.pid or ''} | {experiment.report_dir if experiment.source_ref != 'external' else experiment.external_report_path} | {experiment.log_path if experiment.source_ref != 'external' else experiment.external_log} |"
                )
            lines.append("")

        status_counts: Dict[str, int] = {}
        for experiment in self.experiments.values():
            status_counts[experiment.status] = status_counts.get(experiment.status, 0) + 1
        lines.extend(["## Status Counts", ""])
        for status_name in sorted(status_counts):
            lines.append(f"- {status_name}: {status_counts[status_name]}")
        lines.append("")
        STATE_MD.write_text("\n".join(lines), encoding="utf-8")

    def run(self):
        while not self.stop_requested:
            self.refresh_states()
            launched = False
            while not self.stop_requested:
                free_gpu = self.find_free_gpu()
                if free_gpu is None:
                    break
                next_experiment = self.next_runnable_experiment()
                if next_experiment is None:
                    break
                self.launch_experiment(next_experiment, free_gpu)
                launched = True
                self.persist_state()
            self.persist_state()
            if all(experiment.status in FINAL_STATUSES | {"completed_external", "completed_reused"} for experiment in self.experiments.values()):
                break
            if not launched:
                time.sleep(REFRESH_SECONDS)
        self.refresh_states()
        self.persist_state()


def experiment_name_from_report_dir(report_dir: Path) -> str:
    return report_dir.name


def main():
    scheduler = Scheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
