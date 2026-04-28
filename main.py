from clip.clip import load, tokenize, available_models
import torch
from dataset import *
from torchvision import transforms
import argparse
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import shlex
import sys
import warnings
from util.utils import eval_all_class
import copy

def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)


def uses_prototype_score(args):
    return getattr(args, "score_mode", "clip") == "prototype"


def uses_fg_prompt(clip_model):
    return getattr(clip_model, "fg_prompt_mode", "off") == "on"


def uses_cap_prompt(clip_model):
    return bool(getattr(clip_model, "use_cap_prompt", False)) and getattr(clip_model, "cap_prompt", None) is not None


def coerce_prompt_tensor(loaded_prompt):
    if isinstance(loaded_prompt, torch.nn.Parameter):
        loaded_prompt = loaded_prompt.detach()
    if not torch.is_tensor(loaded_prompt):
        raise TypeError("Expected prompt tensor, got {}".format(type(loaded_prompt)))
    if loaded_prompt.ndim == 2:
        loaded_prompt = loaded_prompt.unsqueeze(0)
    if loaded_prompt.ndim != 3:
        raise ValueError("Expected prompt tensor with shape [K, prompt_len, dim]")
    return loaded_prompt


def match_prompt_embedding(target_prompt, loaded_prompt):
    loaded_prompt = coerce_prompt_tensor(loaded_prompt)
    if isinstance(target_prompt, torch.nn.Parameter):
        target_prompt = target_prompt.detach()
    if target_prompt is None:
        raise ValueError("Target prompt is not initialized")
    if loaded_prompt.shape[1:] != target_prompt.shape[1:]:
        raise ValueError(
            "Prompt shape mismatch: loaded {} vs target {}".format(tuple(loaded_prompt.shape), tuple(target_prompt.shape))
        )
    if loaded_prompt.shape[0] == target_prompt.shape[0]:
        matched_prompt = loaded_prompt
    elif loaded_prompt.shape[0] == 1:
        matched_prompt = loaded_prompt.repeat(target_prompt.shape[0], 1, 1)
    elif target_prompt.shape[0] == 1:
        matched_prompt = loaded_prompt.mean(dim=0, keepdim=True)
    else:
        index = torch.linspace(0, loaded_prompt.shape[0] - 1, steps=target_prompt.shape[0], device=loaded_prompt.device)
        matched_prompt = loaded_prompt.index_select(0, index.round().long())
    return matched_prompt.to(device=target_prompt.device, dtype=target_prompt.dtype)


def extract_shared_prompt_tensor(loaded_prompt):
    if isinstance(loaded_prompt, dict):
        if "state_prompt_embedding" in loaded_prompt:
            return coerce_prompt_tensor(loaded_prompt["state_prompt_embedding"])
        if "prompt_embedding" in loaded_prompt:
            return coerce_prompt_tensor(loaded_prompt["prompt_embedding"])
        if "normal_prompt_embedding" in loaded_prompt and "abnormal_prompt_embedding" in loaded_prompt:
            normal_prompt = coerce_prompt_tensor(loaded_prompt["normal_prompt_embedding"])
            abnormal_prompt = coerce_prompt_tensor(loaded_prompt["abnormal_prompt_embedding"])
            return torch.cat([normal_prompt, abnormal_prompt], dim=0).mean(dim=0, keepdim=True)
        raise ValueError("Unsupported prompt checkpoint format: {}".format(sorted(loaded_prompt.keys())))
    return coerce_prompt_tensor(loaded_prompt)


def assign_prompt_embeddings(clip_model, loaded_prompt):
    if uses_fg_prompt(clip_model):
        if isinstance(loaded_prompt, dict) and "normal_prompt_embedding" in loaded_prompt and "abnormal_prompt_embedding" in loaded_prompt:
            matched_normal_prompt = match_prompt_embedding(
                clip_model.normal_prompt_embedding,
                loaded_prompt["normal_prompt_embedding"],
            )
            matched_abnormal_prompt = match_prompt_embedding(
                clip_model.abnormal_prompt_embedding,
                loaded_prompt["abnormal_prompt_embedding"],
            )
        else:
            shared_prompt = extract_shared_prompt_tensor(loaded_prompt).mean(dim=0, keepdim=True)
            matched_normal_prompt = match_prompt_embedding(clip_model.normal_prompt_embedding, shared_prompt)
            matched_abnormal_prompt = match_prompt_embedding(clip_model.abnormal_prompt_embedding, shared_prompt)
        clip_model.normal_prompt_embedding = torch.nn.Parameter(matched_normal_prompt)
        clip_model.abnormal_prompt_embedding = torch.nn.Parameter(matched_abnormal_prompt)
        return

    matched_prompt = match_prompt_embedding(
        clip_model.state_prompt_embedding,
        extract_shared_prompt_tensor(loaded_prompt),
    )
    clip_model.state_prompt_embedding = torch.nn.Parameter(matched_prompt)


def build_prompt_checkpoint_payload(clip_model):
    if uses_fg_prompt(clip_model):
        return {
            "fg_prompt": "on",
            "num_ab_prompts": int(clip_model.abnormal_prompt_embedding.shape[0]),
            "normal_prompt_embedding": clip_model.normal_prompt_embedding.detach().cpu(),
            "abnormal_prompt_embedding": clip_model.abnormal_prompt_embedding.detach().cpu(),
        }
    return clip_model.state_prompt_embedding.detach().cpu()


def get_trainable_parameter_summary(module):
    trainable_names = []
    trainable_count = 0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        trainable_names.append(name)
        trainable_count += param.numel()
    return trainable_names, trainable_count


def log_trainable_parameter_summary(logger, module, prefix="Trainable parameters"):
    trainable_names, trainable_count = get_trainable_parameter_summary(module)
    logger.info(
        "{}: count={}, names={}".format(
            prefix,
            trainable_count,
            trainable_names,
        )
    )


def log_prompt_configuration(logger, clip_model, args, prefix="Prompt configuration"):
    with torch.no_grad():
        text_features = clip_model.encode_state_prompt(args=args)
    if uses_cap_prompt(clip_model):
        prompt_shapes = {
            "cap_normal_ctx": list(clip_model.cap_prompt.normal_ctx.shape),
            "cap_abnormal_ctx": list(clip_model.cap_prompt.abnormal_ctx.shape),
        }
    elif uses_fg_prompt(clip_model):
        prompt_shapes = {
            "normal_prompt_embedding": list(clip_model.normal_prompt_embedding.shape),
            "abnormal_prompt_embedding": list(clip_model.abnormal_prompt_embedding.shape),
        }
    else:
        prompt_shapes = {
            "state_prompt_embedding": list(clip_model.state_prompt_embedding.shape),
        }
    logger.info(
        "{}: use_cap_prompt={}, fg_prompt={}, num_ab_prompts={}, prompt_len={}, ab_agg={}, cap_abnormal_agg={}, text_feature_shape={}, prompt_shapes={}".format(
            prefix,
            bool(getattr(args, "use_cap_prompt", False)),
            args.fg_prompt,
            args.num_ab_prompts,
            args.prompt_len,
            getattr(args, "ab_agg", "sum_prob"),
            getattr(args, "cap_abnormal_agg", "mean_feature"),
            list(text_features.shape),
            prompt_shapes,
        )
    )


def log_prompt_forward_shapes(logger, clip_model, prefix="Prompt forward"):
    prompt_debug = getattr(clip_model, "_latest_prompt_debug", None)
    if not isinstance(prompt_debug, dict):
        return
    logger.info(
        "{}: prompt_source={}, use_cap_prompt={}, fg_prompt={}, num_ab_prompts={}, ab_agg={}, cap_abnormal_agg={}, prototype_score_active={}, prototype_fusion_applied={}, text_feature_shape={}, scale_text_feature_shape={}, image_cls_logits_shape={}, image_cls_prob_shape={}, pixel_score_shape={}, pixel_prob_shape={}".format(
            prefix,
            prompt_debug.get("prompt_source"),
            prompt_debug.get("use_cap_prompt"),
            prompt_debug.get("fg_prompt"),
            prompt_debug.get("num_ab_prompts"),
            prompt_debug.get("ab_agg"),
            prompt_debug.get("cap_abnormal_agg"),
            prompt_debug.get("prototype_score_active"),
            prompt_debug.get("prototype_fusion_applied"),
            prompt_debug.get("text_feature_shape"),
            prompt_debug.get("scale_text_feature_shape"),
            prompt_debug.get("image_cls_logits_shape"),
            prompt_debug.get("image_cls_prob_shape"),
            prompt_debug.get("pixel_score_shape"),
            prompt_debug.get("pixel_prob_shape"),
        )
    )


def log_cap_configuration(logger, clip_model, args):
    if not uses_cap_prompt(clip_model):
        return
    cap_trainable_names, cap_trainable_count = get_trainable_parameter_summary(clip_model.cap_prompt)
    logger.info(
        "CAP configuration: use_cap_prompt=%s, cap_num_abnormal_prompts=%d, cap_n_normal_ctx=%d, cap_n_abnormal_ctx=%d, lambda_cap_orth=%.6f, cap_abnormal_agg=%s, cap_ctx_init=%s",
        bool(getattr(args, "use_cap_prompt", False)),
        int(getattr(args, "cap_num_abnormal_prompts", 1)),
        int(getattr(args, "cap_n_normal_ctx", 0)),
        int(getattr(args, "cap_n_abnormal_ctx", 0)),
        float(getattr(args, "lambda_cap_orth", 0.0)),
        getattr(args, "cap_abnormal_agg", "mean_feature"),
        getattr(args, "cap_ctx_init", "random"),
    )
    logger.info("CAP trainable parameters: count=%d, names=%s", cap_trainable_count, cap_trainable_names)


def load_training_components(clip_model, args, device, logger=None):
    prompt_path = os.path.join(args.weight, "{}_prompt.pt".format(args.dataset))
    adaptor_path = os.path.join(args.weight, "{}_adaptor.pt".format(args.dataset))
    lsar_path = os.path.join(args.weight, "{}_lsar.pt".format(args.dataset))
    mapb_path = os.path.join(args.weight, "{}_mapb.pt".format(args.dataset))
    cap_prompt_path = os.path.join(args.weight, "{}_cap_prompt.pt".format(args.dataset))

    prompt_state = torch.load(prompt_path, map_location=device)
    assign_prompt_embeddings(clip_model, prompt_state)

    adaptor_state = torch.load(adaptor_path, map_location=device)
    if isinstance(adaptor_state, torch.nn.Module):
        adaptor_state = adaptor_state.state_dict()
    clip_model.adaptor.load_state_dict(adaptor_state)
    clip_model.adaptor = clip_model.adaptor.to(device)

    if getattr(clip_model, "layer_residuals", None) is not None and os.path.exists(lsar_path):
        clip_model.layer_residuals.load_state_dict(torch.load(lsar_path, map_location=device))

    if getattr(clip_model, "prototype_bank", None) is not None and os.path.exists(mapb_path):
        clip_model.prototype_bank.load_state_dict(torch.load(mapb_path, map_location=device))

    if uses_cap_prompt(clip_model):
        if os.path.exists(cap_prompt_path):
            cap_state = torch.load(cap_prompt_path, map_location=device)
            clip_model.cap_prompt.load_state_dict(cap_state)
        else:
            message = "CAP prompt checkpoint not found at {}; using current initialization.".format(cap_prompt_path)
            if logger is not None:
                logger.warning(message)
            else:
                warnings.warn(message)


def save_training_components(clip_model, args, logger):
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        build_prompt_checkpoint_payload(clip_model),
        os.path.join(checkpoint_dir, "{}_prompt.pt".format(args.dataset)),
    )
    torch.save(
        clip_model.adaptor.state_dict(),
        os.path.join(checkpoint_dir, "{}_adaptor.pt".format(args.dataset)),
    )

    if getattr(clip_model, "layer_residuals", None) is not None:
        torch.save(
            clip_model.layer_residuals.state_dict(),
            os.path.join(checkpoint_dir, "{}_lsar.pt".format(args.dataset)),
        )

    if getattr(clip_model, "prototype_bank", None) is not None:
        torch.save(
            clip_model.prototype_bank.state_dict(),
            os.path.join(checkpoint_dir, "{}_mapb.pt".format(args.dataset)),
        )

    if uses_cap_prompt(clip_model):
        torch.save(
            clip_model.cap_prompt.state_dict(),
            os.path.join(checkpoint_dir, "{}_cap_prompt.pt".format(args.dataset)),
        )

    logger.info("Saved checkpoints to %s", checkpoint_dir)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')

def patch_alignment_loss(img_tokens, labels, gts):
    gts = gts.reshape(img_tokens[0].size(0), -1)
    labels = labels.reshape(labels.size(0), 1)
    # labels = torch.cat([labels, gts], dim=1)
    new_gts = copy.copy(gts)
    if(len(new_gts[new_gts == 0])) == 0:
        return 0
    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()
    mask = torch.matmul(new_gts.reshape(b, l, 1), new_gts.reshape(b, 1, l))
    total_sim = 0
    for img_token in img_tokens:
        img_token = img_token[:, 1:, :]
        img_token = torch.nn.functional.normalize(img_token, dim=-1)
        sim = torch.matmul(img_token, img_token.permute(0, 2, 1))
        sim = sim[mask == -1].mean() - sim[mask == 1].mean()
        sim = sim if sim > 0 else 0
        total_sim = total_sim + sim
    return total_sim / len(img_tokens)


def infer_default_mapb_branch_num(args):
    return max(int(getattr(args, "num_ab_prompts", 4) or 4), 1)


def normalize_ab_agg_mode(raw_value):
    mapping = {
        "sum_prob": "sum_prob",
        "sum": "sum_prob",
        "max_prob": "max_prob",
        "max": "max_prob",
        "mean_prob": "mean_prob",
        "mean": "mean_prob",
        "logsumexp_logit": "logsumexp_logit",
        "logsumexp": "logsumexp_logit",
    }
    value = "sum_prob" if raw_value in (None, "") else str(raw_value).strip().lower()
    if value not in mapping:
        raise ValueError("Unsupported ab_agg mode: {}".format(raw_value))
    return mapping[value]


def normalize_cap_agg_mode(raw_value):
    mapping = {
        "mean_feature": "mean_feature",
        "prob_sum": "prob_sum",
        "max_logit": "max_logit",
    }
    value = "mean_feature" if raw_value in (None, "") else str(raw_value).strip().lower()
    if value not in mapping:
        raise ValueError("Unsupported cap_abnormal_agg mode: {}".format(raw_value))
    return mapping[value]


def normalize_runtime_args(args):
    args.use_cap_prompt = bool(getattr(args, "use_cap_prompt", False))
    args.cap_num_abnormal_prompts = max(int(getattr(args, "cap_num_abnormal_prompts", 10)), 1)
    args.cap_n_normal_ctx = max(int(getattr(args, "cap_n_normal_ctx", 4)), 0)
    args.cap_n_abnormal_ctx = max(int(getattr(args, "cap_n_abnormal_ctx", 4)), 0)
    args.cap_abnormal_agg = normalize_cap_agg_mode(getattr(args, "cap_abnormal_agg", "mean_feature"))
    args.cap_log_interval = max(int(getattr(args, "cap_log_interval", 1)), 1)
    requested_prompt_num = max(int(getattr(args, "num_ab_prompts", 4) or 4), 1)
    for legacy_value in [getattr(args, "mapb_branch_num", 0), getattr(args, "mapb_branch_count", 0)]:
        if legacy_value not in (None, 0):
            requested_prompt_num = max(int(legacy_value), 1)
    args.num_ab_prompts = requested_prompt_num
    args.mapb_branch_num = requested_prompt_num
    args.mapb_branch_count = requested_prompt_num
    if args.use_cap_prompt:
        if getattr(args, "fg_prompt", "off") == "on":
            warnings.warn("use_cap_prompt=True overrides legacy fg_prompt; forcing fg_prompt=off.")
        args.fg_prompt = "off"
        args.use_mapb = int(getattr(args, "use_mapb", 0))
        if args.use_mapb and getattr(args, "score_mode", "clip") == "clip":
            args.score_mode = "prototype"
    else:
        if int(getattr(args, "use_mapb", 0)):
            args.fg_prompt = "on"
        args.use_mapb = 1 if getattr(args, "fg_prompt", "off") == "on" else 0
        if args.use_mapb and getattr(args, "score_mode", "clip") == "prototype":
            args.score_mode = "clip"
    args.mapb_default_branch_num = infer_default_mapb_branch_num(args)
    args.mapb_effective_branch_num = args.mapb_default_branch_num
    args.ab_agg = normalize_ab_agg_mode(getattr(args, "ab_agg", "sum_prob"))
    if getattr(args, "mapb_aggregation", None) not in (None, ""):
        args.ab_agg = normalize_ab_agg_mode(getattr(args, "mapb_aggregation"))
    if not hasattr(args, "_command") or args._command is None:
        args._command = " ".join(shlex.quote(token) for token in sys.argv)
    if getattr(args, "debug_mapb", 0) and not getattr(args, "mapb_debug_json", None):
        experiment_name = os.path.basename(os.path.abspath(args.log_dir)) if getattr(args, "log_dir", None) else "mapb_debug"
        args.mapb_debug_json = os.path.join(args.log_dir, f"{experiment_name}_mapb_debug.json")
    return args
    

def train(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = get_logger(os.path.join(args.log_dir, '{}_{}_s{}.txt'.format(args.dataset, args.fewshot, args.seed)))
    print_args(logger, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = load(name=args.model, jit = (not args.model in available_models()), device=device, download_root=args.clip_download_dir)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()
    
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)
    log_prompt_configuration(logger, clip_model, args)
    log_cap_configuration(logger, clip_model, args)
    log_trainable_parameter_summary(logger, clip_model, prefix="Trainable parameter summary")

    dataset_builders = {
        "mvtec": lambda: MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "visa": lambda: VisaDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "btad": lambda: BTADDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "dtd": lambda: DTDDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "dagm": lambda: DAGMDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "isic": lambda: ISICDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "clinic": lambda: ClinicDBDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "colon": lambda: ColonDBDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "brainmri": lambda: BrainMRIDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "br35h": lambda: Br35HDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
        "kvasir": lambda: KvasirDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform),
    }
    if len(args.test_dataset) < 1:
        requested_test_datasets = list(dataset_builders.keys())
    else:
        requested_test_datasets = list(args.test_dataset)

    required_datasets = [args.dataset]
    for ds_name in requested_test_datasets:
        if ds_name not in required_datasets:
            required_datasets.append(ds_name)

    instantiated_datasets = {}
    for ds_name in required_datasets:
        try:
            instantiated_datasets[ds_name] = dataset_builders[ds_name]()
        except FileNotFoundError as exc:
            if ds_name == args.dataset or len(args.test_dataset) > 0:
                raise
            logger.warning("Skip dataset %s because data is unavailable: %s", ds_name, exc)

    test_dataset_dict = {}
    for ds_name in requested_test_datasets:
        if ds_name in instantiated_datasets:
            test_dataset_dict[ds_name] = instantiated_datasets[ds_name]
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]
    train_dataset = instantiated_datasets[args.dataset]
        
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.weight is not None:
        load_training_components(clip_model, args, device, logger=logger)
        log_prompt_configuration(logger, clip_model, args, prefix="Loaded prompt configuration")
        log_cap_configuration(logger, clip_model, args)
    else:
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(), lr=args.lr, betas=(0.5, 0.999))
        first_debug_loss_logged = False
       
        for epoch in range(1, args.epochs + 1):
            total_loss = []
            original_loss_meter = []
            cap_orth_loss_meter = []
            cap_pair_cos_mean_meter = []
            cap_pair_cos_max_meter = []
            cap_pair_cos_min_meter = []
            mapb_proto_loss = []
            mapb_ready_ratio = []
            mapb_fallback_ratio = []
            
            for items in tqdm(train_dataloader):
                imgs, labels, gts = items[:3]
                labels = labels.to(device)
                imgs = imgs.to(device)
                raw_gts = gts.to(device)
                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(imgs, args=args, gts=raw_gts)
                if epoch == 1 and len(total_loss) == 0:
                    log_prompt_forward_shapes(logger, clip_model, prefix="First train forward")
                gts = raw_gts
                gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
                gts[gts < 0.5] = 0
                gts[gts > 0.5] = 1
                
                original_loss = focal_loss(predict_labels, labels) + args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) + args.lambda2 * patch_alignment_loss(img_tokens, labels, gts)
                loss = original_loss
                cap_aux = getattr(clip_model, "_latest_cap_aux", None)
                if args.use_cap_prompt and cap_aux is not None and cap_aux.get("cap_orth_loss") is not None:
                    loss = loss + float(getattr(args, "lambda_cap_orth", 0.0)) * cap_aux["cap_orth_loss"]
                mapb_aux = getattr(clip_model, "_latest_mapb_aux", None)
                if uses_prototype_score(args) and args.lambda_proto > 0 and mapb_aux is not None and mapb_aux.get("prototype_loss_value") is not None:
                    loss = loss + args.lambda_proto * mapb_aux["prototype_loss_value"]
                if bool(getattr(args, "debug_mapb", 0)) and not first_debug_loss_logged:
                    logger.info(
                        "[MAPB DEBUG] first_batch_loss=%.6f requested_branch_num=%s effective_branch_num=%s",
                        float(loss.detach().item()),
                        getattr(args, "mapb_branch_num", None),
                        getattr(args, "mapb_effective_branch_num", None),
                    )
                    first_debug_loss_logged = True
                optimizer.zero_grad()
                
                
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                original_loss_meter.append(float(original_loss.detach().item()))
                if args.use_cap_prompt and cap_aux is not None:
                    cap_orth_loss_meter.append(float(cap_aux["cap_orth_loss"].detach().item()))
                    cap_pair_cos_mean_meter.append(float(cap_aux.get("cap_abn_pair_cos_mean", 0.0)))
                    cap_pair_cos_max_meter.append(float(cap_aux.get("cap_abn_pair_cos_max", 0.0)))
                    cap_pair_cos_min_meter.append(float(cap_aux.get("cap_abn_pair_cos_min", 0.0)))
                if uses_prototype_score(args) and mapb_aux is not None:
                    if mapb_aux.get("prototype_loss") is not None:
                        mapb_proto_loss.append(float(mapb_aux["prototype_loss"].detach().item()))
                    if mapb_aux.get("prototype_ready_ratio") is not None:
                        mapb_ready_ratio.append(float(mapb_aux["prototype_ready_ratio"].detach().item()))
                    if mapb_aux.get("prototype_fallback_ratio") is not None:
                        mapb_fallback_ratio.append(float(mapb_aux["prototype_fallback_ratio"].detach().item()))
            if uses_prototype_score(args):
                logger.info(
                    "Epoch: {}/{}, Original_loss: {:.6f}, Total_loss: {:.6f}, MAPB_loss: {:.6f}, MAPB_ready_ratio: {:.6f}, MAPB_fallback_ratio: {:.6f}".format(
                        epoch,
                        args.epochs,
                        np.mean(original_loss_meter) if len(original_loss_meter) > 0 else 0.0,
                        np.mean(total_loss),
                        np.mean(mapb_proto_loss) if len(mapb_proto_loss) > 0 else 0.0,
                        np.mean(mapb_ready_ratio) if len(mapb_ready_ratio) > 0 else 0.0,
                        np.mean(mapb_fallback_ratio) if len(mapb_fallback_ratio) > 0 else 0.0,
                    )
                )
            else:
                logger.info(
                    "Epoch: {}/{}, Original_loss: {:.6f}, Total_loss: {:.6f}".format(
                        epoch,
                        args.epochs,
                        np.mean(original_loss_meter) if len(original_loss_meter) > 0 else 0.0,
                        np.mean(total_loss),
                    )
                )
            if args.use_cap_prompt and epoch % args.cap_log_interval == 0:
                logger.info(
                    "CAP Epoch: {}/{}, CAP_orth_loss: {:.6f}, lambda_cap_orth: {:.6f}, CAP_pair_cos_mean: {:.6f}, CAP_pair_cos_max: {:.6f}, CAP_pair_cos_min: {:.6f}".format(
                        epoch,
                        args.epochs,
                        np.mean(cap_orth_loss_meter) if len(cap_orth_loss_meter) > 0 else 0.0,
                        float(getattr(args, "lambda_cap_orth", 0.0)),
                        np.mean(cap_pair_cos_mean_meter) if len(cap_pair_cos_mean_meter) > 0 else 0.0,
                        np.mean(cap_pair_cos_max_meter) if len(cap_pair_cos_max_meter) > 0 else 0.0,
                        np.mean(cap_pair_cos_min_meter) if len(cap_pair_cos_min_meter) > 0 else 0.0,
                    )
                )
        save_training_components(clip_model, args, logger)
    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info("---------------------------{}------------------------------".format(dataset_name))
        eval_all_class(clip_model, dataset_name, test_ds, args, logger, device)
        logger.info("-------------------------------------------------------------")

      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--clip_download_dir', type=str, default='./download/clip/', help='training dataset')
    
    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')
    
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='model')
    
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning tate')
    
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')
    
    parser.add_argument('--epochs', type=int, default=2, help='training epoch')
    
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')

    parser.add_argument('--fg_prompt', type=str, default='off', choices=['off', 'on'], help='legacy text-side multi-abnormal prompt branch; kept for backward compatibility and superseded by CAP when --use_cap_prompt is enabled')

    parser.add_argument('--num_ab_prompts', type=int, default=4, help='number of abnormal prompts for the legacy fg_prompt branch')

    parser.add_argument('--use_cap_prompt', action='store_true', default=False, help='enable CAP-style text-side compound abnormality prompt learning')

    parser.add_argument('--cap_num_abnormal_prompts', type=int, default=10, help='number of complementary abnormal prompts in CAP')

    parser.add_argument('--cap_n_normal_ctx', type=int, default=4, help='number of shared normal context tokens in CAP')

    parser.add_argument('--cap_n_abnormal_ctx', type=int, default=4, help='number of abnormal-specific context tokens per CAP prompt')

    parser.add_argument('--cap_ctx_init', type=str, default='random', choices=['random', 'template'], help='initialization mode for CAP learnable context tokens')

    parser.add_argument('--lambda_cap_orth', type=float, default=0.01, help='loss weight for CAP abnormal-prompt orthogonal regularization')

    parser.add_argument('--cap_abnormal_agg', type=str, default='mean_feature', choices=['mean_feature', 'prob_sum', 'max_logit'], help='aggregation rule for CAP abnormal text prompts')

    parser.add_argument('--cap_log_interval', type=int, default=1, help='log CAP diagnostics every N epochs')
    
    parser.add_argument('--category', type=str, default=None, help='normal class')
    
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    
    parser.add_argument('--seed', type=int, default=122, help='seed')
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    
    parser.add_argument('--img_size', type=int, default=518)
    
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24], help='choose vit layers to extract features')
    
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')
    
    parser.add_argument('--weight', type=str, default=None, help='load weight path')
    
    parser.add_argument('--vis', type=int, default=0, help='visualization results')
    
    parser.add_argument('--vis_dir', type=str, default='./vis_results/', help='visualization results dir')
    
    parser.add_argument('--memory_layers',  nargs='+', type=int, default=[6, 12, 18, 24], help='choose resnet layers to store and compare features')
    
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')
    
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')

    parser.add_argument('--use_lsar', type=int, default=0, help='enable layer-specific adaptor residuals after adaptor projection')

    parser.add_argument('--lsar_bottleneck_ratio', type=int, default=4, help='bottleneck ratio for layer-specific adaptor residuals')

    parser.add_argument('--lsar_zero_init', type=int, default=1, help='zero-initialize the last LSAR projection for residual start-up')

    parser.add_argument('--use_mvti', type=int, default=0, help='enable horizontal-flip multi-view test-time inference')

    parser.add_argument('--mvti_views', type=int, default=2, help='number of multi-view test-time transforms when MVTI is enabled')

    parser.add_argument('--use_mapb', type=int, default=0, help='legacy MAPB compatibility switch; when CAP is enabled it activates the visual prototype-bank path without replacing CAP text scoring')

    parser.add_argument('--score_mode', type=str, default='clip', choices=['clip', 'prototype'], help='score computation mode; prototype activates the optional prototype-bank path')

    parser.add_argument('--lambda_proto', type=float, default=0.1, help='loss weight for MAPB prototype-bank objective')

    parser.add_argument('--prototype_k', type=int, default=4, help='number of MAPB prototypes per branch')

    parser.add_argument('--prototype_momentum', type=float, default=0.95, help='EMA momentum for MAPB prototype updates')

    parser.add_argument('--prototype_temperature', type=float, default=0.07, help='temperature for MAPB prototype contrastive loss')

    parser.add_argument('--prototype_max_samples', type=int, default=4096, help='maximum normal patch tokens sampled per branch for MAPB updates')

    parser.add_argument('--prototype_fusion_alpha', type=float, default=0.25, help='fusion weight between CLIP anomaly map and MAPB anomaly map')

    parser.add_argument('--mapb_branch_num', type=int, default=0, help='deprecated alias for num_ab_prompts; kept for compatibility with older scripts')

    parser.add_argument('--mapb_branch_count', type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument('--mapb_aggregation', type=str, default=None, help=argparse.SUPPRESS)

    parser.add_argument('--ab_agg', type=str, default='sum_prob', choices=['sum_prob', 'max_prob', 'mean_prob', 'logsumexp_logit'], help='aggregation rule for the legacy fg_prompt multi-abnormal text branch')

    parser.add_argument('--dump_prompt_diag_json', type=str, default='', help='optional json path for prompt diagnostics during evaluation; supports both legacy fg_prompt and CAP')

    parser.add_argument('--debug_mapb', type=int, default=0, help='print and save MAPB debug configuration without changing training outputs')

    parser.add_argument('--mapb_debug_json', type=str, default=None, help='optional JSON path for MAPB debug dump when debug_mapb=1')
    
    
    args = parser.parse_args()
    args = normalize_runtime_args(args)
    args.seed = setup_seed(args.seed)
    train(args)
    
    
    
