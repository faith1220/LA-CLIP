#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT_DIR / "reports" / "prcv_laclip_mapb_debug"
DIAG_ROOT = REPORT_ROOT / "diagnostics"
AUTO_SUMMARY_CSV = ROOT_DIR / "reports" / "prcv_laclip_auto" / "experiment_summary.csv"
RUNNER_SUMMARY_CSV = REPORT_ROOT / "MAPB_CLEAN_ABLATION.csv"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    if value in (None, "", "None"):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_code_audit(diag_payloads: Dict[str, Dict]) -> str:
    default_payload = diag_payloads["default"]
    mc = default_payload["model_mapb_config"]
    pc = default_payload["prompt_config"]
    ff = default_payload["first_forward_shapes"]
    lines = [
        "# MAPB Code Audit",
        "",
        "## Relevant Files",
        "",
        "- `main.py`: CLI definition, runtime branch-number normalization, debug-mapb argument handling, and training-side MAPB loss logging.",
        "- `clip/model.py`: MAPB insertion, effective branch-number derivation, aggregated image-token branches, prototype-bank construction, prompt/text encoding, and runtime debug JSON dump.",
        "- `clip/prototype_bank.py`: hyperspherical prototype bank over branch-wise image tokens; no abnormal text-branch implementation is present here.",
        "- `scripts/prcv_laclip_mapb_debug/diagnose_mapb_config.py`: minimal model+batch diagnosis entrypoint.",
        "",
        "## CLI Parameters",
        "",
        "- `--use_mapb`: enables the MAPB prototype-bank path.",
        "- `--score_mode {clip,prototype}`: `prototype` activates the prototype-bank scoring path; `main.py` still couples `use_mapb=1` and `score_mode=prototype`.",
        "- `--mapb_branch_num`: requested MAPB prototype branch count. `0` means use the default branch count derived from `feature_layers`.",
        "- `--mapb_aggregation {mean,max,logsumexp}`: aggregation rule over branch-wise MAPB anomaly maps.",
        "- `--debug_mapb`: enables verbose MAPB config logging and JSON dump.",
        "",
        "## Default Logic",
        "",
        f"- Default `feature_layers`: `{mc['feature_layers']}`.",
        f"- Fixed branches per feature layer: `{mc['branches_per_layer']}` from neighbor radii `[1, 3, 5]`.",
        f"- Therefore the default effective MAPB branch count is `{mc['default_branch_num']}`.",
        f"- `branch_num=0` does **not** mean zero abnormal branches. It means `use default={mc['default_branch_num']}` aggregated prototype branches.",
        "",
        "## What MAPB Branch Count Actually Controls",
        "",
        f"- Effective branch count is stored in `mapb_effective_branch_num={mc['effective_branch_num']}` for the default payload.",
        f"- The prototype bank shape under default is `{mc['prototype_bank_shape']}` = `[num_branches, prototype_k, dim]`.",
        "- The branch count controls the number of **aggregated image-token prototype branches** only.",
        "- It does **not** control the number of abnormal text prompts.",
        "",
        "## Prompt / Text Prototype Path",
        "",
        f"- Normal prompt: `{pc['normal_prompt']}`.",
        f"- Abnormal prompts: `{pc['abnormal_prompts']}`.",
        f"- Abnormal prompt count: `{pc['abnormal_prompt_count']}`.",
        f"- Text prompt order: `{pc['text_prototype_order']}`.",
        f"- Learnable context token shape: `{pc['learnable_context_shape']}`.",
        f"- State prompt token shape: `{default_payload['text_feature_shapes']['state_prompt_tokens']}`.",
        f"- First-forward text feature shape: `{ff['text_feature_shape']}`.",
        "- Conclusion: the current clean MAPB implementation has exactly one abnormal text prompt and two text prototypes `[normal, abnormal]` regardless of `mapb_branch_num`.",
        "",
        "## Logit / Aggregation Path",
        "",
        f"- First-forward branch logit shapes under default: `{ff['branch_logit_shapes']}`.",
        f"- Normal logit shape: `{ff['normal_logits_shape']}`.",
        f"- Abnormal logit shape: `{ff['abnormal_logits_shape']}`.",
        f"- Final abnormal probability map shape: `{ff['final_abnormal_probability_shape']}`.",
        f"- Aggregation type under default diagnosis: `{mc['aggregation_type']}`.",
        "",
        "## Audit Conclusions",
        "",
        f"1. `branch_num=0` maps to the default effective branch count `{mc['default_branch_num']}`, not zero branches.",
        "2. `mapb_branch_num` changes prototype-bank branches over aggregated image tokens, not abnormal text-branch count.",
        "3. Prompt template, text prototype count, and score path stay unchanged when only `mapb_branch_num` changes.",
        "4. The old 'abnormal branch number' ablation was semantically mislabeled for the current clean MAPB implementation.",
        "",
    ]
    return "\n".join(lines)


def build_problem_diagnosis(diag_payloads: Dict[str, Dict], old_rows: List[Dict[str, str]]) -> str:
    default_payload = diag_payloads["default"]
    default_num = default_payload["model_mapb_config"]["default_branch_num"]
    old_map = {row["experiment_name"]: row for row in old_rows if row["experiment_name"].startswith("mapb_branch_") or row["experiment_name"] in {"comp_mapb_lsar", "main_m2v_full", "comp_full"}}
    lines = [
        "# MAPB Problem Diagnosis",
        "",
        "## Answers",
        "",
        f"1. `branch_num=0/default` actually means `effective_branch_num={default_num}`.",
        "2. Explicit `branch_num=1/2/4/6` does really change the MAPB branch count, but it changes the number of prototype-bank branches over aggregated image tokens, not the number of abnormal prompts.",
        "3. Explicit `branch_num` does not change prompt text content. The clean implementation always uses `without defect.` and `with defect.`.",
        "4. Explicit `branch_num` does not change aggregation type by itself. In the diagnosis payloads, aggregation stays `mean` for all default/1/2/4/6/8 settings.",
        "5. Explicit `branch_num` does not switch score mode. The diagnosis payloads show `score_mode=prototype` throughout.",
        "6. Explicit `branch_num` did not trigger MAPB fallback in diagnosis. Ready ratio stayed `1.0` and fallback ratio stayed `0.0` for the tested settings.",
        "7. Explicit `branch_num` did not change text feature shape. The text feature shape stayed `[2, 768]` because the text prototypes are still `[normal, abnormal]`.",
        "8. Explicit `branch_num` did not create a logit-dimension error. The abnormal logit path stayed valid; only the number of branch-wise prototype banks changed.",
        "9. `mapb_branch_num` and `prototype_k` are independent. The branch-number change modifies the first dimension of the bank (`num_branches`), while `prototype_k` controls the second dimension (`num_prototypes per branch`).",
        "10. The old high `branch_num=0` result corresponds to the default full MAPB setting and was effectively reusing the full/default branch configuration, not a true '0 branch' setting.",
        "11. The old auto scheduler metadata was misleading because it recorded the requested branch argument (`0`) instead of the effective default branch count (`12`).",
        "12. Root cause classification: `default special logic + metadata error + semantic mislabeling`. It is not a prompt-template change, aggregation-path change, or prototype-score change.",
        "",
        "## Evidence",
        "",
        f"- Default diagnostic JSON shows `requested_branch_num=0`, `default_branch_num={default_num}`, `effective_branch_num={default_num}`, `abnormal_prompt_count=1`, `prototype_bank_shape=[{default_num}, 4, 768]`.",
        "- Explicit diagnostic JSONs show:",
    ]
    for key in ["1", "2", "4", "6", "8"]:
        payload = diag_payloads[key]
        mc = payload["model_mapb_config"]
        pc = payload["prompt_config"]
        ff = payload["first_forward_shapes"]
        lines.append(
            f"  - `branch_num={key}` -> `effective_branch_num={mc['effective_branch_num']}`, "
            f"`abnormal_prompt_count={pc['abnormal_prompt_count']}`, `text_feature_shape={ff['text_feature_shape']}`, "
            f"`aggregation={mc['aggregation_type']}`, `score_mode={mc['score_mode']}`."
        )
    lines.extend(
        [
            "",
            "## Why the Previous Branch Ablation Looked Abnormal",
            "",
            "- The old branch ablation was interpreted as changing the number of abnormal MAPB branches.",
            "- In reality, it was shrinking the prototype-bank branch count from the default full setting (`12`) down to `1/2/4/6`.",
            "- Therefore the `branch_num=0` row was not a weak setting. It was the strongest full/default setting, which explains why it dominated the explicit small-number rows.",
            "",
            "## Old Numeric Evidence",
            "",
        ]
    )
    if old_map:
        lines.extend(
            [
                "| Experiment | Recorded branch_num | I-AUROC | I-AP | P-AUROC | P-AP | PRO |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ["comp_mapb_lsar", "comp_full", "main_m2v_full", "mapb_branch_1", "mapb_branch_2", "mapb_branch_4", "mapb_branch_6"]:
            row = old_map.get(name)
            if not row:
                continue
            lines.append(
                f"| {name} | {row.get('branch_num', '')} | {row.get('i_auroc', '')} | {row.get('i_ap', '')} | {row.get('p_auroc', '')} | {row.get('p_ap', '')} | {row.get('pro', '')} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_final_report(diag_payloads: Dict[str, Dict], rerun_rows: List[Dict[str, str]]) -> str:
    default_num = diag_payloads["default"]["model_mapb_config"]["default_branch_num"]
    completed_rows = [row for row in rerun_rows if row.get("status") == "completed" and row["experiment_name"].startswith("mapb_clean_branch_") or row["experiment_name"] == "mapb_clean_default_explicit"]
    completed_rows = [row for row in rerun_rows if row.get("status") == "completed" and (row["experiment_name"].startswith("mapb_clean_branch_") or row["experiment_name"] == "mapb_clean_default_explicit")]
    best_row = None
    if completed_rows:
        best_row = max(completed_rows, key=lambda row: float(row.get("i_auroc") or -1))
    lines = [
        "# MAPB Debug Report",
        "",
        "## 1. Problem Phenomenon",
        "",
        "- In the previous automatic ablation, `branch_num=1/2/4/6` was much worse than the row recorded as `branch_num=0/default`.",
        "- That behavior looked suspicious because a nominally 'zero-branch' setting should not outperform all explicit branch settings.",
        "",
        "## 2. Code Audit Conclusion",
        "",
        "- MAPB is implemented in `clip/model.py` + `clip/prototype_bank.py`, with CLI wiring in `main.py`.",
        f"- In the current clean codebase, `branch_num=0` means `use default effective branch count = {default_num}`.",
        "- The branch argument controls prototype-bank branches over aggregated image tokens, not abnormal text-prompt count.",
        "- The clean text prompt path remains `[normal, abnormal]` for all tested branch settings.",
        "",
        "## 3. Diagnosis Summary",
        "",
        "| Setting | Requested | Effective | Abnormal Prompt Count | Text Feature Shape | Aggregation | Score Mode |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for key in ["default", "1", "2", "4", "6", "8"]:
        payload = diag_payloads[key]
        mc = payload["model_mapb_config"]
        pc = payload["prompt_config"]
        ff = payload["first_forward_shapes"]
        lines.append(
            f"| {key} | {mc['requested_branch_num']} | {mc['effective_branch_num']} | {pc['abnormal_prompt_count']} | {ff['text_feature_shape']} | {mc['aggregation_type']} | {mc['score_mode']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Root Cause",
            "",
            "- The anomaly was caused by a combination of `default special logic + misleading metadata + semantic mislabeling`.",
            f"- The old `branch_num=0` row was actually the default full setting with `effective_branch_num={default_num}`.",
            "- The explicit `1/2/4/6` rows were genuine low-branch prototype-bank ablations, so they were naturally weaker than the full/default setting.",
            "",
            "## 5. Fixes Applied",
            "",
            "- Added `--debug_mapb` and JSON dump support to record requested/effective branch counts, prompt config, prototype-bank shape, and first-forward shapes.",
            "- Added `scripts/prcv_laclip_mapb_debug/diagnose_mapb_config.py` for minimal MAPB configuration diagnosis.",
            "- Clarified `mapb_branch_num` semantics in runtime logging: `0 means using default=<actual_num>`.",
            "- Fixed the old auto scheduler summary logic so branch metadata can be reported as effective branch count rather than raw requested `0`.",
            "",
            "## 6. Clean Rerun Setting",
            "",
            "- Source dataset: `mvtec`",
            "- Target dataset: `visa`",
            "- Seed: `122`",
            "- Epochs: `2` for clean reruns (`1` for smoke checks)",
            "- Batch size: `8` for clean reruns (`2` for smoke checks)",
            "- Fixed modules: `use_mapb=1`, `use_lsar=1`, `lsar_bottleneck_ratio=4`, `use_mvti=0`, `score_mode=prototype`",
            "- Fixed prototype settings: `prototype_k=4`, `prototype_momentum=0.95`, `prototype_temperature=0.07`, `prototype_max_samples=4096`, `prototype_fusion_alpha=0.25`",
            f"- The only intended variable is the effective MAPB prototype-branch count: `1/2/4/6/8/{default_num}`.",
            "",
            "## 7. Clean Result Table",
            "",
        ]
    )
    if rerun_rows:
        lines.extend(
            [
                "| Experiment | Req Branch | Eff Branch | Aggregation | Status | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | PRO |",
                "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rerun_rows:
            if not (row["experiment_name"].startswith("mapb_clean_branch_") or row["experiment_name"].startswith("mapb_clean_agg_") or row["experiment_name"] == "mapb_clean_default_explicit"):
                continue
            lines.append(
                f"| {row['experiment_name']} | {row['requested_branch_num']} | {row['effective_branch_num']} | {row['aggregation']} | {row['status']} | "
                f"{row.get('i_auroc', '')} | {row.get('i_ap', '')} | {row.get('i_f1', '')} | {row.get('p_auroc', '')} | {row.get('p_ap', '')} | {row.get('pro', '')} |"
            )
    lines.extend(
        [
            "",
            "## 8. Recommendation",
            "",
            f"- Default explicit branch count is `{default_num}`.",
            f"- Best completed clean branch row by I-AUROC: `{best_row['experiment_name']}`" if best_row else "- Best row is not available yet because clean reruns have not all completed.",
            f"- Recommended paper setting: `effective_branch_num={best_row['effective_branch_num']}`." if best_row else f"- Recommended paper setting will be chosen between `{default_num}` and the best clean rerun row after completion.",
            "- This ablation is now semantically clean because the branch count is explicitly reported as an effective prototype-branch count.",
            "",
            "## 9. Risks",
            "",
            "- This is still a single-seed ablation, so ranking stability across seeds is not yet verified.",
            "- If branch settings are close, at least one additional seed is recommended before a strong causal claim.",
            "- Because the current clean MAPB does not implement multiple abnormal text branches, this study should be described as a `prototype-branch count` ablation, not an `abnormal prompt branch` ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_clean_ablation_tables(rerun_rows: List[Dict[str, str]]):
    filtered_rows = [
        row
        for row in rerun_rows
        if row["experiment_name"].startswith("mapb_clean_branch_")
        or row["experiment_name"] == "mapb_clean_default_explicit"
        or row["experiment_name"].startswith("mapb_clean_agg_")
    ]
    fieldnames = [
        "experiment_name",
        "requested_branch_num",
        "effective_branch_num",
        "abnormal_prompt_count",
        "aggregation",
        "score_mode",
        "prototype_k",
        "prototype_fusion_alpha",
        "seed",
        "i_auroc",
        "i_ap",
        "i_f1",
        "p_auroc",
        "p_ap",
        "pro",
        "status",
        "log_path",
        "report_path",
        "debug_json_path",
    ]
    with (REPORT_ROOT / "MAPB_CLEAN_ABLATION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# MAPB Clean Ablation",
        "",
        "| Experiment | Req Branch | Eff Branch | Aggregation | Status | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | PRO |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in filtered_rows:
        lines.append(
            f"| {row['experiment_name']} | {row.get('requested_branch_num', '')} | {row.get('effective_branch_num', '')} | {row.get('aggregation', '')} | {row.get('status', '')} | "
            f"{row.get('i_auroc', '')} | {row.get('i_ap', '')} | {row.get('i_f1', '')} | {row.get('p_auroc', '')} | {row.get('p_ap', '')} | {row.get('pro', '')} |"
        )
    (REPORT_ROOT / "MAPB_CLEAN_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    diag_payloads = {
        "default": load_json(DIAG_ROOT / "mapb_branch_default_debug.json"),
        "1": load_json(DIAG_ROOT / "mapb_branch_1_debug.json"),
        "2": load_json(DIAG_ROOT / "mapb_branch_2_debug.json"),
        "4": load_json(DIAG_ROOT / "mapb_branch_4_debug.json"),
        "6": load_json(DIAG_ROOT / "mapb_branch_6_debug.json"),
        "8": load_json(DIAG_ROOT / "mapb_branch_8_debug.json"),
    }
    old_rows = load_csv_rows(AUTO_SUMMARY_CSV)
    rerun_rows = load_csv_rows(RUNNER_SUMMARY_CSV)

    (DIAG_ROOT / "mapb_code_audit.md").write_text(build_code_audit(diag_payloads), encoding="utf-8")
    (DIAG_ROOT / "mapb_problem_diagnosis.md").write_text(build_problem_diagnosis(diag_payloads, old_rows), encoding="utf-8")
    (REPORT_ROOT / "MAPB_DEBUG_REPORT.md").write_text(build_final_report(diag_payloads, rerun_rows), encoding="utf-8")
    write_clean_ablation_tables(rerun_rows)


if __name__ == "__main__":
    main()
