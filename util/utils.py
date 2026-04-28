import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from statistics import mean
from numpy import ndarray
from skimage import measure
import pandas as pd
from scipy.ndimage import gaussian_filter
from clip.model import CLIP
from torchvision import transforms
import torch
import numpy as np
from dataset.mvtec import MVTecDataset
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from torchvision import transforms
from clip.model import CLIP
import copy
from torchvision import models
import math
import matplotlib.pyplot as plt
import seaborn as sns
import json

def transform_invert(img_, transform_train):
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None]) 
    return img_


def show_cam_on_image(img, anomaly_map, alpha=0.5):
    img = np.float32(img)
    anomaly_map = np.float32(anomaly_map)
    cam  = alpha * img + (1 - alpha) * anomaly_map
    return np.uint8(cam)



def cvt2heatmap(gray):
    gray = np.float32(gray)
    gray = normalize(gray)
    gray = gray * 255
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = normalize(scoremap)
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def plot_attention(attention_weights, filename, vmax=None):
    """
    """
    nrows, ncols = attention_weights.shape[0], attention_weights.shape[1]
    
    for row in range(nrows):
        for col in range(ncols):
            fig, ax = plt.subplots(figsize=(10, 5))
           
            im = ax.imshow(attention_weights[row, col], 
                         cmap='viridis', 
                         interpolation='nearest',
                         vmax=vmax
                         )
            ax.axis('off')  
            file_path = f"{filename}_{row}_{col}.png"
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True,)
            plt.close()
            
def visualize(clip_model:CLIP, test_dataset, args, transform, device):
    cnt = 0
    with torch.no_grad():
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        cnt = 0
        for data in test_dataloader:
            img_paths = data[-1]
            labels = data[1]
            if torch.sum(labels) >= 1:
                imgs = data[0].to(device)
                _, anomaly_maps = clip_model.detect_forward(imgs, args)
                anomaly_maps = F.interpolate(anomaly_maps, size=(imgs.size(-2), imgs.size(-1)), mode='bilinear').cpu().numpy()
                anomaly_maps = np.stack([gaussian_filter(mask, sigma=4) for mask in anomaly_maps])
                anomaly_maps = anomaly_maps.reshape(anomaly_maps.shape[0], anomaly_maps.shape[2],  anomaly_maps.shape[3])
                imgs = transform_invert(imgs, transform)
                gts = data[2].squeeze()
                if len(gts.shape) == 3:
                    pack = zip(imgs, anomaly_maps, gts, labels, img_paths)
                else:
                    pack = zip(imgs, anomaly_maps, labels, img_paths)
                for p in pack:
                    if p[-2] != 0:
                        print(p[-1])
                        save_file_name = '_'.join(p[-1].split('/')[-2:])
                        ano_map = cvt2heatmap(p[1])
                        img = cv2.cvtColor((p[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cam_map = show_cam_on_image(img, ano_map)
                        result_path = os.path.join(args.vis_dir, '{}-shot'.format(args.fewshot), test_dataset.dataset_name, test_dataset.category)
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        if len(p) == 5:
                            gt = cvt2heatmap(p[2])
                            cam_gt = show_cam_on_image(img, gt)
                            res = np.concatenate((img, cam_gt, cam_map), axis=1)
                        else:
                            res = np.vstack((img, ano_map, cam_map))
                        img_path = os.path.join(result_path, save_file_name)
                        cv2.imwrite(img_path, res)
                        cnt += 1

def calculate_metrics(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
    max_f1 = np.max(f1_scores)
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return {"AUROC": roc, "AP": ap, 'max-F1': max_f1}


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        # score_res_str += '\n'
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v) 
    return score_res_str

def cal_average_res(total_res):
    avergae = {}
    category_num = len(total_res)
    for res in total_res: # every category res
        for key, ip in res.items(): # sample or pixel
            if key not in avergae:
                avergae[key] = {}
            for m, v in ip.items():
                if m not in avergae[key]:
                    avergae[key][m] = 0
                avergae[key][m] += v
    
    for key, ip in avergae.items():
        for m, v in ip.items():
            avergae[key][m] = v / category_num
    
    return avergae
    

def detect_single_view(clip_model, imgs, args):
    predict_labels, predict_masks = clip_model.detect_forward(imgs, args)
    return predict_labels, predict_masks


def get_mvti_transforms(num_views):
    num_views = max(int(num_views), 1)

    transforms_list = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        (lambda x: torch.flip(x, dims=[-2, -1]), lambda x: torch.flip(x, dims=[-2, -1])),
        (lambda x: x.transpose(-1, -2), lambda x: x.transpose(-1, -2)),
        (lambda x: torch.flip(x.transpose(-1, -2), dims=[-1]), lambda x: torch.flip(x, dims=[-1]).transpose(-1, -2)),
        (lambda x: torch.flip(x.transpose(-1, -2), dims=[-2]), lambda x: torch.flip(x, dims=[-2]).transpose(-1, -2)),
        (lambda x: torch.flip(x.transpose(-1, -2), dims=[-2, -1]), lambda x: torch.flip(x, dims=[-2, -1]).transpose(-1, -2)),
    ]
    return transforms_list[: min(num_views, len(transforms_list))]


def multi_view_inference(clip_model, imgs, args):
    mvti_views = int(getattr(args, "mvti_views", 2))
    labels_per_view = []
    masks_per_view = []
    for forward_fn, inverse_fn in get_mvti_transforms(mvti_views):
        view_imgs = forward_fn(imgs)
        view_labels, view_masks = detect_single_view(clip_model, view_imgs, args)
        labels_per_view.append(view_labels)
        masks_per_view.append(inverse_fn(view_masks))
    return torch.stack(labels_per_view, dim=0).mean(dim=0), torch.stack(masks_per_view, dim=0).mean(dim=0)

def _init_prompt_diag_accumulator():
    return {
        "ab_agg": None,
        "abnormal_text_features": None,
        "update_calls": 0,
        "image_prob_sum": None,
        "image_prob_sq_sum": None,
        "image_winner_count": None,
        "image_count": 0,
        "pixel_prob_sum": None,
        "pixel_prob_sq_sum": None,
        "pixel_winner_count": None,
        "pixel_count": 0,
    }


def _update_prompt_diag_accumulator(accumulator, analysis):
    if accumulator is None or not isinstance(analysis, dict):
        return
    accumulator["update_calls"] += 1

    abnormal_text_features = analysis.get("abnormal_text_features")
    image_abnormal_probs = analysis.get("image_abnormal_probs")
    pixel_abnormal_probs = analysis.get("pixel_abnormal_probs")

    if torch.is_tensor(abnormal_text_features) and accumulator["abnormal_text_features"] is None:
        accumulator["abnormal_text_features"] = abnormal_text_features.detach().float().cpu().clone()

    if image_abnormal_probs is not None and torch.is_tensor(image_abnormal_probs):
        image_abnormal_probs = image_abnormal_probs.detach().float().cpu()
        num_channels = image_abnormal_probs.shape[-1]
        if accumulator["image_prob_sum"] is None:
            accumulator["image_prob_sum"] = torch.zeros(num_channels, dtype=torch.float32)
            accumulator["image_prob_sq_sum"] = torch.zeros(num_channels, dtype=torch.float32)
            accumulator["image_winner_count"] = torch.zeros(num_channels, dtype=torch.float32)
        accumulator["image_prob_sum"] += image_abnormal_probs.sum(dim=0)
        accumulator["image_prob_sq_sum"] += image_abnormal_probs.square().sum(dim=0)
        winners = image_abnormal_probs.argmax(dim=-1)
        accumulator["image_winner_count"] += torch.bincount(winners, minlength=num_channels).to(torch.float32)
        accumulator["image_count"] += int(image_abnormal_probs.shape[0])

    if pixel_abnormal_probs is not None and torch.is_tensor(pixel_abnormal_probs):
        pixel_abnormal_probs = pixel_abnormal_probs.detach().float().cpu()
        flat_pixel_probs = pixel_abnormal_probs.reshape(-1, pixel_abnormal_probs.shape[-1])
        num_channels = flat_pixel_probs.shape[-1]
        if accumulator["pixel_prob_sum"] is None:
            accumulator["pixel_prob_sum"] = torch.zeros(num_channels, dtype=torch.float32)
            accumulator["pixel_prob_sq_sum"] = torch.zeros(num_channels, dtype=torch.float32)
            accumulator["pixel_winner_count"] = torch.zeros(num_channels, dtype=torch.float32)
        accumulator["pixel_prob_sum"] += flat_pixel_probs.sum(dim=0)
        accumulator["pixel_prob_sq_sum"] += flat_pixel_probs.square().sum(dim=0)
        winners = flat_pixel_probs.argmax(dim=-1)
        accumulator["pixel_winner_count"] += torch.bincount(winners, minlength=num_channels).to(torch.float32)
        accumulator["pixel_count"] += int(flat_pixel_probs.shape[0])

    if accumulator["ab_agg"] is None:
        accumulator["ab_agg"] = analysis.get("ab_agg")


def _summarize_prompt_channel_usage(prob_sum, prob_sq_sum, winner_count, total_count, level_name):
    if prob_sum is None or prob_sq_sum is None or winner_count is None or total_count < 1:
        return []
    mean_prob = prob_sum / float(total_count)
    variance = torch.clamp(prob_sq_sum / float(total_count) - mean_prob.square(), min=0.0)
    std_prob = torch.sqrt(variance)
    winner_ratio = winner_count / float(total_count)
    rows = []
    for channel_idx in range(mean_prob.numel()):
        rows.append(
            {
                "level": level_name,
                "channel_idx": channel_idx + 1,
                "mean_prob": float(mean_prob[channel_idx].item()),
                "std_prob": float(std_prob[channel_idx].item()),
                "top1_winner_ratio": float(winner_ratio[channel_idx].item()),
            }
        )
    return rows


def _finalize_prompt_diag_accumulator(accumulator):
    if accumulator is None or accumulator.get("abnormal_text_features") is None:
        return None

    abnormal_text_features = accumulator["abnormal_text_features"]
    normalized_abnormal = F.normalize(abnormal_text_features, dim=-1)
    similarity_matrix = torch.matmul(normalized_abnormal, normalized_abnormal.t()).cpu()
    offdiag_mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
    offdiag_values = similarity_matrix[offdiag_mask]
    if offdiag_values.numel() > 0:
        offdiag_stats = {
            "mean": float(offdiag_values.mean().item()),
            "max": float(offdiag_values.max().item()),
            "min": float(offdiag_values.min().item()),
        }
    else:
        offdiag_stats = {"mean": float("nan"), "max": float("nan"), "min": float("nan")}

    return {
        "ab_agg": accumulator.get("ab_agg"),
        "update_calls": int(accumulator.get("update_calls", 0)),
        "num_ab_prompts": int(similarity_matrix.shape[0]),
        "prompt_similarity_matrix": similarity_matrix.tolist(),
        "prompt_similarity_offdiag": offdiag_stats,
        "channel_usage": _summarize_prompt_channel_usage(
            accumulator.get("image_prob_sum"),
            accumulator.get("image_prob_sq_sum"),
            accumulator.get("image_winner_count"),
            accumulator.get("image_count", 0),
            "image",
        )
        + _summarize_prompt_channel_usage(
            accumulator.get("pixel_prob_sum"),
            accumulator.get("pixel_prob_sq_sum"),
            accumulator.get("pixel_winner_count"),
            accumulator.get("pixel_count", 0),
            "pixel",
        ),
    }


def _dump_prompt_diag_json(json_path, payload):
    if not json_path:
        return
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluation_pixel(clip_model:CLIP, dataset_name, dataloader, args, device, logger=None):
    pixel_gt_list = []
    pixel_score_list = []
    sample_gt_list = []
    sample_score_list = []
    aupro_list = []
    res = {}
    pro = 0
    with torch.no_grad():
        for items in tqdm(dataloader):
            imgs, labels, gt = items[:3]
            imgs = imgs.to(device)
            if getattr(args, "use_mvti", 0):
                predict_labels, predict_masks = multi_view_inference(clip_model, imgs, args)
            else:
                predict_labels, predict_masks = detect_single_view(clip_model, imgs, args)
            prompt_diag_accumulator = getattr(clip_model, "_prompt_diag_accumulator", None)
            if prompt_diag_accumulator is not None:
                _update_prompt_diag_accumulator(
                    prompt_diag_accumulator,
                    getattr(clip_model, "_latest_prompt_analysis", None),
                )
            
            predict_masks = F.interpolate(predict_masks, size=(imgs.size(-2), imgs.size(-1)), mode='bilinear').cpu().numpy()
            predict_masks = np.stack([gaussian_filter(mask, sigma=4) for mask in predict_masks])
            sample_gt_list.append(labels)
            sample_score_list.append(predict_labels.cpu().numpy())
            
            if dataset_name not in ['br35h', 'brainmri', 'headct']:
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                gt = gt.cpu().numpy().astype(int)
                
                labels = np.max(gt.reshape(gt.shape[0], -1), axis=-1)
                pixel_gt_list.append(gt)
                pixel_score_list.append(predict_masks)
        if dataset_name not in ['isic', 'clinic', 'colon', 'kvasir', 'endo']:                
            sample_gt_list = np.concatenate(sample_gt_list)
            sample_score_list = np.concatenate(sample_score_list)
            res['Sample_CLS'] = calculate_metrics(sample_score_list, sample_gt_list)
        if dataset_name not in ['br35h', 'brainmri', 'headct']:
            pixel_gt_list = np.concatenate(pixel_gt_list)
            pixel_score_list = np.concatenate(pixel_score_list)
            if len(pixel_gt_list.shape) == 4:
                pixel_gt_list = pixel_gt_list.squeeze(1)
            if len(pixel_score_list.shape) == 4:
                pixel_score_list = pixel_score_list.squeeze(1)
            pro = compute_pro(pixel_gt_list, pixel_score_list)
            
            res['Pixel'] = calculate_metrics(pixel_score_list.reshape(-1), pixel_gt_list.reshape(-1))
            res['Pixel']['PRO'] = pro
            
    return res



def eval_all_class(clip_model: CLIP, dataset_name, test_dataset, args, logger, device):
    total_res = []
    logger.info(
        "Evaluation prompt mode: use_cap_prompt=%s, fg_prompt=%s, cap_abnormal_agg=%s, ab_agg=%s",
        bool(getattr(args, "use_cap_prompt", False)),
        getattr(args, "fg_prompt", "off"),
        getattr(args, "cap_abnormal_agg", "mean_feature"),
        getattr(args, "ab_agg", "sum_prob"),
    )
    collect_prompt_diag = bool(getattr(args, "dump_prompt_diag_json", "")) and (
        bool(getattr(args, "use_cap_prompt", False))
        or getattr(clip_model, "fg_prompt_mode", "off") == "on"
    )
    clip_model._prompt_diag_accumulator = _init_prompt_diag_accumulator() if collect_prompt_diag else None
    if args.fewshot > 0:
        fewshot_dataset = copy.deepcopy(test_dataset)
        fewshot_dataset.train = True
        fewshot_dataset.fewshot = args.fewshot
        
    for category in test_dataset.categories:
        print(category)
        test_dataset.update(category)
        
        ## store memory
        if args.fewshot > 0:
            print("use few shot")
            fewshot_dataset.update(category)
            logger.info("{}, {}".format(fewshot_dataset.category, fewshot_dataset.cur_img_paths))
            few_shot_dataloader = torch.utils.data.DataLoader(fewshot_dataset, batch_size=max(args.fewshot, args.batch_size), shuffle=False)
            with torch.no_grad():
                for items in tqdm(few_shot_dataloader):
                    imgs = items[0].to(device)
                    # imgs = torch.cat([imgs] + [transforms.functional.rotate(imgs, 90) for degrees in [90, 180, 270]])       
                    clip_model.store_memory(imgs, args)
                
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        if args.vis != 0:
            print("visualize")
            # visualize_attention_map(clip_model, test_dataset, args, test_dataset.transform, device)
            visualize(clip_model, test_dataset, args, test_dataset.transform, device)
        else:
            category_res = evaluation_pixel(clip_model, dataset_name, test_dataloader, args, device, logger=logger)
            total_res.append(category_res)
            res_str = get_res_str(category_res)
            logger.info("Category {}: {}".format(category, res_str))
    if args.vis == 0:
        average_res = cal_average_res(total_res)
        average_res_str = get_res_str(average_res)
        logger.info("Average: {}".format(average_res_str))
        if collect_prompt_diag:
            prompt_diag_payload = _finalize_prompt_diag_accumulator(getattr(clip_model, "_prompt_diag_accumulator", None))
            if prompt_diag_payload is None:
                fallback_accumulator = _init_prompt_diag_accumulator()
                _update_prompt_diag_accumulator(fallback_accumulator, getattr(clip_model, "_latest_prompt_analysis", None))
                prompt_diag_payload = _finalize_prompt_diag_accumulator(fallback_accumulator)
            if prompt_diag_payload is not None:
                prompt_diag_payload.update(
                    {
                        "source_dataset": getattr(args, "dataset", ""),
                        "target_dataset": dataset_name,
                        "use_cap_prompt": bool(getattr(args, "use_cap_prompt", False)),
                        "fg_prompt": getattr(args, "fg_prompt", "off"),
                        "num_ab_prompts": int(getattr(args, "num_ab_prompts", 1)),
                        "cap_num_abnormal_prompts": int(getattr(args, "cap_num_abnormal_prompts", 1)),
                        "cap_abnormal_agg": getattr(args, "cap_abnormal_agg", "mean_feature"),
                        "seed": int(getattr(args, "seed", 0)),
                        "log_dir": getattr(args, "log_dir", ""),
                    }
                )
                _dump_prompt_diag_json(getattr(args, "dump_prompt_diag_json", ""), prompt_diag_payload)
                offdiag = prompt_diag_payload["prompt_similarity_offdiag"]
                logger.info(
                    "Prompt diagnostics [{}]: updates={}, offdiag_cos mean={:.6f}, max={:.6f}, min={:.6f}".format(
                        dataset_name,
                        int(prompt_diag_payload.get("update_calls", 0)),
                        float(offdiag["mean"]),
                        float(offdiag["max"]),
                        float(offdiag["min"]),
                    )
                )
            else:
                logger.warning("Prompt diagnostics [{}]: no prompt-analysis batches were captured".format(dataset_name))
        clip_model._prompt_diag_accumulator = None
        
