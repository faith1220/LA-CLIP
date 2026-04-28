#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from clip.clip import available_models, load, tokenize
from dataset import MVTecDataset, VisaDataset
from main import load_training_components, normalize_runtime_args, setup_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose MAPB configuration without full training")
    parser.add_argument("--clip_download_dir", type=str, default="./download/clip")
    parser.add_argument("--data_dir", type=str, default="./data_local")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--test_dataset", nargs="+", type=str, default=["visa"])
    parser.add_argument("--model", type=str, default="ViT-L/14@336px")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--feature_layers", nargs="+", type=int, default=[6, 12, 18, 24])
    parser.add_argument("--memory_layers", nargs="+", type=int, default=[6, 12, 18, 24])
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--prompt_len", type=int, default=12)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="defect")
    parser.add_argument("--vis", type=int, default=0)
    parser.add_argument("--vis_dir", type=str, default="./vis_results/")
    parser.add_argument("--use_lsar", type=int, default=1)
    parser.add_argument("--lsar_bottleneck_ratio", type=int, default=4)
    parser.add_argument("--lsar_zero_init", type=int, default=1)
    parser.add_argument("--use_mvti", type=int, default=0)
    parser.add_argument("--mvti_views", type=int, default=2)
    parser.add_argument("--use_mapb", type=int, default=1)
    parser.add_argument("--score_mode", type=str, default="prototype", choices=["clip", "prototype"])
    parser.add_argument("--lambda_proto", type=float, default=0.1)
    parser.add_argument("--prototype_k", type=int, default=4)
    parser.add_argument("--prototype_momentum", type=float, default=0.95)
    parser.add_argument("--prototype_temperature", type=float, default=0.07)
    parser.add_argument("--prototype_max_samples", type=int, default=4096)
    parser.add_argument("--prototype_fusion_alpha", type=float, default=0.25)
    parser.add_argument("--mapb_branch_num", type=str, default="default")
    parser.add_argument("--mapb_aggregation", type=str, default="mean", choices=["mean", "max", "logsumexp"])
    parser.add_argument("--debug_mapb", type=int, default=1)
    parser.add_argument("--mapb_debug_json", type=str, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    return parser


def build_source_dataset(name, root, clip_transform, target_transform):
    builders = {
        "mvtec": lambda: MVTecDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "visa": lambda: VisaDataset(root=root, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
    }
    return builders[name]()


def main():
    parser = build_parser()
    args = parser.parse_args()
    branch_setting = str(args.mapb_branch_num).strip().lower()
    if branch_setting == "default":
        args.mapb_branch_num = 0
    else:
        args.mapb_branch_num = int(branch_setting)
    args.mapb_debug_json = args.out_json
    args.log_dir = args.log_dir or str(Path(args.out_json).resolve().parent)
    args._command = " ".join([token if " " not in token else f"'{token}'" for token in sys.argv])
    args.seed = setup_seed(args.seed)
    args = normalize_runtime_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = load(
        name=args.model,
        jit=(args.model not in available_models()),
        device=device,
        download_root=args.clip_download_dir,
    )
    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose(
        [
            transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)
    if args.weight is not None:
        load_training_components(clip_model, args, device)

    dataset = build_source_dataset(args.dataset, args.data_dir, clip_transform, target_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(dataloader))
    imgs, _, gts = batch[:3]
    imgs = imgs.to(device)
    gts = gts.to(device)
    with torch.no_grad():
        clip_model.detect_forward_seg(imgs, args=args, gts=gts)

    payload_path = Path(args.out_json)
    if not payload_path.exists():
        payload = {
            "experiment_name": payload_path.stem,
            "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
            "model_mapb_config": clip_model._mapb_debug_model_config["model_mapb_config"],
            "prompt_config": clip_model._mapb_debug_model_config["prompt_config"],
            "text_feature_shapes": clip_model._mapb_debug_model_config["text_feature_shapes"],
            "aggregation_config": clip_model._mapb_debug_model_config["aggregation_config"],
            "first_forward_shapes": clip_model._mapb_debug_first_forward,
        }
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    final_payload = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
