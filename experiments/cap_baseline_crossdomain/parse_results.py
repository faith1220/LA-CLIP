#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "mvtec_to_visa_A0_baseline",
        "direction": "MVTec -> VisA",
        "train_dataset": "mvtec",
        "test_dataset": "visa",
        "method": "A0 baseline",
    },
    {
        "name": "mvtec_to_visa_A1_cap",
        "direction": "MVTec -> VisA",
        "train_dataset": "mvtec",
        "test_dataset": "visa",
        "method": "A1 baseline + CAP",
    },
    {
        "name": "visa_to_mvtec_A0_baseline",
        "direction": "VisA -> MVTec",
        "train_dataset": "visa",
        "test_dataset": "mvtec",
        "method": "A0 baseline",
    },
    {
        "name": "visa_to_mvtec_A1_cap",
        "direction": "VisA -> MVTec",
        "train_dataset": "visa",
        "test_dataset": "mvtec",
        "method": "A1 baseline + CAP",
    },
]

METRIC_PATTERNS = {
    "I_AUROC": r"Sample_CLS_AUROC:\s*([0-9.]+)",
    "I_AP": r"Sample_CLS_AP:\s*([0-9.]+)",
    "I_F1": r"Sample_CLS_max-F1:\s*([0-9.]+)",
    "P_AUROC": r"Pixel_AUROC:\s*([0-9.]+)",
    "P_AP": r"Pixel_AP:\s*([0-9.]+)",
    "PRO": r"Pixel_PRO:\s*([0-9.]+)",
}

CAP_PATTERNS = {
    "CAP_orth_loss": r"CAP_orth_loss:\s*([-+0-9.eE]+)",
    "CAP_pair_cos_mean": r"CAP_pair_cos_mean:\s*([-+0-9.eE]+)",
    "CAP_pair_cos_max": r"CAP_pair_cos_max:\s*([-+0-9.eE]+)",
    "CAP_pair_cos_min": r"CAP_pair_cos_min:\s*([-+0-9.eE]+)",
    "prompt_source": r"prompt_source=([^,\s]+)",
    "prototype_fusion_applied": r"prototype_fusion_applied=([^,\s]+)",
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_last_float(text: str, pattern: str):
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_last_token(text: str, pattern: str):
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    return str(matches[-1]).strip()


def format_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}"


def load_status_map(status_tsv: Path):
    status = {}
    if not status_tsv.exists():
        return status
    with status_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            status[row["experiment"]] = row
    return status


def parse_experiment(exp_root: Path, meta: dict, status_map: dict):
    name = meta["name"]
    ckpt_dir = exp_root / "ckpts" / name
    tee_log = exp_root / "logs" / f"{name}.log"
    internal_logs = sorted(ckpt_dir.glob("*.txt"))
    prompt_diag = exp_root / "reports" / f"{name}_prompt_diag.json"

    text_parts = [read_text(tee_log)]
    text_parts.extend(read_text(path) for path in internal_logs)
    combined_text = "\n".join(part for part in text_parts if part)

    row = {
        "direction": meta["direction"],
        "train_dataset": meta["train_dataset"],
        "test_dataset": meta["test_dataset"],
        "method": meta["method"],
        "experiment_name": name,
        "I_AUROC": None,
        "I_AP": None,
        "I_F1": None,
        "P_AUROC": None,
        "P_AP": None,
        "PRO": None,
        "log_path": str(tee_log),
        "ckpt_path": str(ckpt_dir),
        "internal_log_path": str(internal_logs[-1]) if internal_logs else "",
        "prompt_diag_json": str(prompt_diag) if prompt_diag.exists() else "",
        "status": "not_started",
        "exit_code": "",
        "CAP_orth_loss": None,
        "CAP_pair_cos_mean": None,
        "CAP_pair_cos_max": None,
        "CAP_pair_cos_min": None,
        "prompt_source": "",
        "prototype_fusion_applied": "",
    }

    for key, pattern in METRIC_PATTERNS.items():
        row[key] = parse_last_float(combined_text, pattern)

    if name.endswith("_A1_cap"):
        for key in ["CAP_orth_loss", "CAP_pair_cos_mean", "CAP_pair_cos_max", "CAP_pair_cos_min"]:
            row[key] = parse_last_float(combined_text, CAP_PATTERNS[key])
        row["prompt_source"] = parse_last_token(combined_text, CAP_PATTERNS["prompt_source"])
        row["prototype_fusion_applied"] = parse_last_token(combined_text, CAP_PATTERNS["prototype_fusion_applied"])
        if prompt_diag.exists():
            try:
                payload = json.loads(read_text(prompt_diag))
                offdiag = payload.get("prompt_similarity_offdiag", {})
                if row["CAP_pair_cos_mean"] is None:
                    row["CAP_pair_cos_mean"] = offdiag.get("mean")
                if row["CAP_pair_cos_max"] is None:
                    row["CAP_pair_cos_max"] = offdiag.get("max")
                if row["CAP_pair_cos_min"] is None:
                    row["CAP_pair_cos_min"] = offdiag.get("min")
            except json.JSONDecodeError:
                pass

    status_row = status_map.get(name)
    if status_row is not None:
        row["exit_code"] = status_row.get("exit_code", "")
        if status_row.get("exit_code", "") == "0":
            has_metrics = any(row[key] is not None for key in METRIC_PATTERNS)
            row["status"] = "completed" if has_metrics else "completed_no_metrics"
        else:
            row["status"] = "failed"
    elif combined_text:
        has_metrics = any(row[key] is not None for key in METRIC_PATTERNS)
        row["status"] = "completed" if has_metrics else "incomplete"

    return row


def write_summary_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "direction",
        "train_dataset",
        "test_dataset",
        "method",
        "I_AUROC",
        "I_AP",
        "I_F1",
        "P_AUROC",
        "P_AP",
        "PRO",
        "log_path",
        "ckpt_path",
        "status",
        "exit_code",
        "internal_log_path",
        "prompt_diag_json",
        "CAP_orth_loss",
        "CAP_pair_cos_mean",
        "CAP_pair_cos_max",
        "CAP_pair_cos_min",
        "prompt_source",
        "prototype_fusion_applied",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = row.copy()
            for key in ["I_AUROC", "I_AP", "I_F1", "P_AUROC", "P_AP", "PRO", "CAP_orth_loss", "CAP_pair_cos_mean", "CAP_pair_cos_max", "CAP_pair_cos_min"]:
                if serializable[key] is None:
                    serializable[key] = ""
            writer.writerow(serializable)


def delta(cap_row, base_row, key):
    if cap_row is None or base_row is None:
        return None
    cap_value = cap_row.get(key)
    base_value = base_row.get(key)
    if cap_value is None or base_value is None:
        return None
    return cap_value - base_value


def markdown_result_table(rows):
    header = "| Method | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | PRO | Status |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    body = [header, sep]
    for row in rows:
        body.append(
            "| {method} | {I_AUROC} | {I_AP} | {I_F1} | {P_AUROC} | {P_AP} | {PRO} | {status} |".format(
                method=row["method"],
                I_AUROC=format_float(row["I_AUROC"]),
                I_AP=format_float(row["I_AP"]),
                I_F1=format_float(row["I_F1"]),
                P_AUROC=format_float(row["P_AUROC"]),
                P_AP=format_float(row["P_AP"]),
                PRO=format_float(row["PRO"]),
                status=row["status"],
            )
        )
    return "\n".join(body)


def markdown_delta_lines(base_row, cap_row):
    keys = ["I_AUROC", "I_AP", "I_F1", "P_AUROC", "P_AP", "PRO"]
    labels = {
        "I_AUROC": "ΔI-AUROC",
        "I_AP": "ΔI-AP",
        "I_F1": "ΔI-F1",
        "P_AUROC": "ΔP-AUROC",
        "P_AP": "ΔP-AP",
        "PRO": "ΔPRO",
    }
    parts = []
    for key in keys:
        value = delta(cap_row, base_row, key)
        if value is None:
            parts.append(f"{labels[key]}: ")
        else:
            parts.append(f"{labels[key]}: {value:+.6f}")
    return "\n".join(f"- {item}" for item in parts)


def write_summary_md(rows, out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    by_direction = {}
    for row in rows:
        by_direction.setdefault(row["direction"], []).append(row)

    lines = ["# CAP Baseline Cross-Domain Results", ""]
    for direction in ["MVTec -> VisA", "VisA -> MVTec"]:
        direction_rows = by_direction.get(direction, [])
        if not direction_rows:
            continue
        lines.append(f"## {direction}")
        lines.append("")
        lines.append(markdown_result_table(direction_rows))
        lines.append("")
        baseline_row = next((row for row in direction_rows if row["method"] == "A0 baseline"), None)
        cap_row = next((row for row in direction_rows if row["method"] == "A1 baseline + CAP"), None)
        lines.append("CAP vs baseline delta:")
        lines.append(markdown_delta_lines(baseline_row, cap_row))
        lines.append("")

    cap_rows = [row for row in rows if row["method"] == "A1 baseline + CAP"]
    lines.append("## CAP Diagnostics")
    lines.append("")
    lines.append("| Direction | CAP_orth_loss | CAP_pair_cos_mean | CAP_pair_cos_max | CAP_pair_cos_min | prompt_source | prototype_fusion_applied |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in cap_rows:
        lines.append(
            "| {direction} | {CAP_orth_loss} | {CAP_pair_cos_mean} | {CAP_pair_cos_max} | {CAP_pair_cos_min} | {prompt_source} | {prototype_fusion_applied} |".format(
                direction=row["direction"],
                CAP_orth_loss=format_float(row["CAP_orth_loss"]),
                CAP_pair_cos_mean=format_float(row["CAP_pair_cos_mean"]),
                CAP_pair_cos_max=format_float(row["CAP_pair_cos_max"]),
                CAP_pair_cos_min=format_float(row["CAP_pair_cos_min"]),
                prompt_source=row["prompt_source"],
                prototype_fusion_applied=row["prototype_fusion_applied"],
            )
        )
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_failed_md(rows, out_md: Path, failed_file: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Failed CAP Baseline Runs", ""]
    failed_rows = [row for row in rows if row["status"] != "completed"]
    if not failed_rows and not failed_file.exists():
        lines.append("No failed runs.")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("| Experiment | Status | Exit Code | Log Path | Checkpoint Path |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in failed_rows:
        lines.append(
            "| {experiment_name} | {status} | {exit_code} | {log_path} | {ckpt_path} |".format(
                **row
            )
        )
    if failed_file.exists():
        raw_failed = failed_file.read_text(encoding="utf-8", errors="ignore").strip()
        if raw_failed:
            lines.append("")
            lines.append("Raw failed_runs.txt:")
            lines.append("")
            lines.append("```text")
            lines.append(raw_failed)
            lines.append("```")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Parse CAP baseline cross-domain experiment results.")
    parser.add_argument("--exp-root", type=str, default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    exp_root = Path(args.exp_root).resolve()
    report_root = exp_root / "reports"
    status_map = load_status_map(report_root / "run_status.tsv")

    rows = [parse_experiment(exp_root, meta, status_map) for meta in EXPERIMENTS]
    write_summary_csv(rows, report_root / "results_summary.csv")
    write_summary_md(rows, report_root / "results_summary.md")
    write_failed_md(rows, report_root / "failed_runs.md", exp_root / "failed_runs.txt")

    completed = sum(1 for row in rows if row["status"] == "completed")
    failed = sum(1 for row in rows if row["status"] != "completed")
    print(f"Parsed {len(rows)} experiments: completed={completed}, non_completed={failed}")
    print(f"CSV: {report_root / 'results_summary.csv'}")
    print(f"Markdown: {report_root / 'results_summary.md'}")


if __name__ == "__main__":
    main()
