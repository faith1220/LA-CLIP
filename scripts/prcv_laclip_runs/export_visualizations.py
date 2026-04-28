#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clip.clip import available_models, load, tokenize
from dataset import ClinicDBDataset, ColonDBDataset, KvasirDataset, MVTecDataset, VisaDataset
from main import load_training_components
from util.utils import detect_single_view, multi_view_inference


def build_dataset(name, root, clip_transform, target_transform):
    builders = {
        "mvtec": lambda: MVTecDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "visa": lambda: VisaDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "clinic": lambda: ClinicDBDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "colon": lambda: ColonDBDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "kvasir": lambda: KvasirDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
    }
    return builders[name]()


def collect_samples(dataset, sample_limit):
    samples = []
    for category in dataset.categories:
        dataset.update(category)
        for idx in range(len(dataset)):
            item = dataset[idx]
            gt = item[2]
            gt_np = np.array(gt if isinstance(gt, Image.Image) else gt.squeeze().numpy())
            if gt_np.max() <= 0:
                continue
            samples.append(
                {
                    "category": category,
                    "path": item[-1],
                    "gt_np": (gt_np * 255).astype(np.uint8) if gt_np.max() <= 1.0 else gt_np.astype(np.uint8),
                }
            )
            break
        if len(samples) >= sample_limit:
            break
    return samples


def build_model(args, device):
    clip_model, clip_transform = load(
        name=args.model,
        jit=(args.model not in available_models()),
        device=device,
        download_root=args.clip_download_dir,
    )
    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)
    return clip_model, clip_transform


def run_map(clip_model, img_tensor, args):
    with torch.no_grad():
        if getattr(args, "use_mvti", 0):
            _, predict_masks = multi_view_inference(clip_model, img_tensor, args)
        else:
            _, predict_masks = detect_single_view(clip_model, img_tensor, args)
    predict_masks = F.interpolate(predict_masks, size=(img_tensor.size(-2), img_tensor.size(-1)), mode="bilinear").cpu().numpy()
    predict_masks = np.stack([gaussian_filter(mask, sigma=4) for mask in predict_masks])
    return predict_masks[0, 0]


def save_heatmap_panel(output_base, image_np, gt_np, method_maps):
    panels = [("image", image_np), ("gt", gt_np)]
    value_maps = [value for _, value in method_maps]
    vmax = max(float(np.max(item)) for item in value_maps) if value_maps else 1.0
    for method_name, method_map in method_maps:
        panels.append((method_name, method_map))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (title, panel) in zip(axes, panels):
        axis.set_title(title)
        axis.axis("off")
        if title == "image":
            axis.imshow(image_np)
        elif title == "gt":
            axis.imshow(gt_np, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(panel, cmap="jet", vmin=0.0, vmax=vmax)
    plt.tight_layout()
    plt.savefig(f"{output_base}.png", dpi=200)
    plt.savefig(f"{output_base}.pdf")
    plt.close(fig)


def make_args(namespace, source_dataset, target_dataset, weight_dir, use_mapb, use_lsar, use_mvti, score_mode, mvti_views=2):
    args = argparse.Namespace(
        model=namespace.model,
        clip_download_dir=namespace.clip_download_dir,
        data_dir=namespace.data_dir,
        dataset=source_dataset,
        test_dataset=[target_dataset],
        batch_size=1,
        epochs=0,
        seed=namespace.seed,
        log_dir=str(namespace.output_dir),
        suffix="defect",
        img_size=namespace.img_size,
        feature_layers=[6, 12, 18, 24],
        weight=str(weight_dir),
        vis=0,
        vis_dir=str(namespace.output_dir),
        memory_layers=[6, 12, 18, 24],
        lambda1=1,
        lambda2=1,
        use_lsar=use_lsar,
        lsar_bottleneck_ratio=4,
        lsar_zero_init=1,
        use_mvti=use_mvti,
        mvti_views=mvti_views,
        use_mapb=use_mapb,
        score_mode=score_mode,
        lambda_proto=0.1,
        prototype_k=4,
        prototype_momentum=0.95,
        prototype_temperature=0.07,
        prototype_max_samples=4096,
        prototype_fusion_alpha=0.25,
        mapb_branch_count=0,
        mapb_aggregation="mean",
        prompt_len=12,
        category=None,
        fewshot=0,
        alpha=0.1,
        lr=0.0001,
    )
    return args


def main():
    parser = argparse.ArgumentParser(description="Export PRCV visualization panels")
    parser.add_argument("--data_dir", type=str, default="./data_local")
    parser.add_argument("--clip_download_dir", type=str, default="./download/clip")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, default="mvtec")
    parser.add_argument("--target_dataset", type=str, default="visa")
    parser.add_argument("--baseline_weight", type=str, default="")
    parser.add_argument("--mapb_lsar_weight", type=str, default="")
    parser.add_argument("--full_weight", type=str, default="")
    parser.add_argument("--model", type=str, default="ViT-L/14@336px")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--sample_limit", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "cross_domain_heatmap_comparison": "partial",
        "mapb_multi_branch_response": "unsupported",
        "lsar_layer_response": "unsupported",
        "mvti_stabilization": "partial",
        "notes": [],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bootstrap_args = make_args(args, args.source_dataset, args.target_dataset, args.baseline_weight or args.mapb_lsar_weight or args.full_weight, 0, 0, 0, "clip")
    _, clip_transform = build_model(bootstrap_args, device=device)
    target_transform = transforms.Compose(
        [
            transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    dataset = build_dataset(args.target_dataset, args.data_dir, clip_transform, target_transform)
    samples = collect_samples(dataset, args.sample_limit)
    if not samples:
        status["notes"].append("No anomalous target samples found for visualization export.")
        (output_dir / "visualization_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return

    method_specs = []
    if args.baseline_weight:
        method_specs.append(("baseline", args.baseline_weight, 0, 0, 0, "clip", 1))
    else:
        status["notes"].append("Baseline checkpoint unavailable; baseline panel skipped.")
    if args.mapb_lsar_weight:
        method_specs.append(("mapb_lsar", args.mapb_lsar_weight, 1, 1, 0, "prototype", 1))
    else:
        status["notes"].append("MAPB+LSAR checkpoint unavailable; middle panel skipped.")
    if args.full_weight:
        method_specs.append(("full", args.full_weight, 1, 1, 1, "prototype", 2))
    else:
        status["notes"].append("Full checkpoint unavailable; full panel skipped.")

    comparison_ready = len(method_specs) >= 2
    mvti_ready = bool(args.full_weight)

    for method_name, weight_dir, use_mapb, use_lsar, use_mvti, score_mode, mvti_views in method_specs:
        method_args = make_args(args, args.source_dataset, args.target_dataset, weight_dir, use_mapb, use_lsar, use_mvti, score_mode, mvti_views=mvti_views)
        clip_model, _ = build_model(method_args, device=device)
        load_training_components(clip_model, method_args, device)
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            img_tensor = clip_transform(image).unsqueeze(0).to(device)
            anomaly_map = run_map(clip_model, img_tensor, method_args)
            sample_id = Path(sample["path"]).stem
            np.save(output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_{method_name}.npy", anomaly_map)
        del clip_model
        if device == "cuda":
            torch.cuda.empty_cache()

    for sample in samples:
        image = Image.open(sample["path"]).convert("RGB")
        image_np = np.asarray(image)
        gt_np = sample["gt_np"]
        sample_id = Path(sample["path"]).stem
        method_maps = []
        for method_name, _, _, _, _, _, _ in method_specs:
            map_path = output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_{method_name}.npy"
            if map_path.exists():
                method_maps.append((method_name, np.load(map_path)))
        if method_maps:
            base_path = output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_comparison"
            save_heatmap_panel(str(base_path), image_np, gt_np, method_maps)

        if mvti_ready:
            single_view_args = make_args(args, args.source_dataset, args.target_dataset, args.full_weight, 1, 1, 0, "prototype", mvti_views=1)
            multi_view_args = make_args(args, args.source_dataset, args.target_dataset, args.full_weight, 1, 1, 1, "prototype", mvti_views=2)
            full_model, _ = build_model(single_view_args, device=device)
            load_training_components(full_model, single_view_args, device)
            img_tensor = clip_transform(image).unsqueeze(0).to(device)
            single_view_map = run_map(full_model, img_tensor, single_view_args)
            mvti_map = run_map(full_model, img_tensor, multi_view_args)
            np.save(output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_single_view.npy", single_view_map)
            np.save(output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_mvti.npy", mvti_map)
            save_heatmap_panel(
                str(output_dir / f"{args.source_dataset}_to_{args.target_dataset}_{sample['category']}_{sample_id}_mvti"),
                image_np,
                gt_np,
                [("single_view", single_view_map), ("mvti_final", mvti_map)],
            )
            del full_model
            if device == "cuda":
                torch.cuda.empty_cache()

    if comparison_ready:
        status["cross_domain_heatmap_comparison"] = "completed"
    if mvti_ready:
        status["mvti_stabilization"] = "completed"
    status["notes"].append("MAPB branch-level and LSAR layer-level intermediate maps are not exposed by the current clean inference API, so these panels are left unsupported.")
    (output_dir / "visualization_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
