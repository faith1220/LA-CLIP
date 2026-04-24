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
    return bool(getattr(args, "use_mapb", 0)) or getattr(args, "score_mode", "clip") == "prototype"


def load_training_components(clip_model, args, device):
    prompt_path = os.path.join(args.weight, "{}_prompt.pt".format(args.dataset))
    adaptor_path = os.path.join(args.weight, "{}_adaptor.pt".format(args.dataset))
    lsar_path = os.path.join(args.weight, "{}_lsar.pt".format(args.dataset))
    mapb_path = os.path.join(args.weight, "{}_mapb.pt".format(args.dataset))

    prompt_state = torch.load(prompt_path, map_location=device)
    if isinstance(prompt_state, torch.nn.Parameter):
        prompt_state = prompt_state.detach()
    clip_model.state_prompt_embedding = torch.nn.Parameter(prompt_state.to(device))
    clip_model.state_prompt_embedding.requires_grad_(True)

    adaptor_state = torch.load(adaptor_path, map_location=device)
    if isinstance(adaptor_state, torch.nn.Module):
        adaptor_state = adaptor_state.state_dict()
    clip_model.adaptor.load_state_dict(adaptor_state)
    clip_model.adaptor = clip_model.adaptor.to(device)

    if getattr(clip_model, "layer_residuals", None) is not None and os.path.exists(lsar_path):
        clip_model.layer_residuals.load_state_dict(torch.load(lsar_path, map_location=device))

    if getattr(clip_model, "prototype_bank", None) is not None and os.path.exists(mapb_path):
        clip_model.prototype_bank.load_state_dict(torch.load(mapb_path, map_location=device))


def save_training_components(clip_model, args, logger):
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        clip_model.state_prompt_embedding.detach().cpu(),
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
        load_training_components(clip_model, args, device)
    else:
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(), lr=args.lr, betas=(0.5, 0.999))
       
        for epoch in range(1, args.epochs + 1):
            total_loss = []
            mapb_proto_loss = []
            mapb_ready_ratio = []
            mapb_fallback_ratio = []
            
            for items in tqdm(train_dataloader):
                imgs, labels, gts = items[:3]
                labels = labels.to(device)
                imgs = imgs.to(device)
                raw_gts = gts.to(device)
                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(imgs, args=args, gts=raw_gts)
                gts = raw_gts
                gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
                gts[gts < 0.5] = 0
                gts[gts > 0.5] = 1
                
                loss = focal_loss(predict_labels, labels) + args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) + args.lambda2 * patch_alignment_loss(img_tokens, labels, gts) 
                mapb_aux = getattr(clip_model, "_latest_mapb_aux", None)
                if uses_prototype_score(args) and args.lambda_proto > 0 and mapb_aux is not None and mapb_aux.get("prototype_loss_value") is not None:
                    loss = loss + args.lambda_proto * mapb_aux["prototype_loss_value"]
                optimizer.zero_grad()
                
                
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                if uses_prototype_score(args) and mapb_aux is not None:
                    if mapb_aux.get("prototype_loss") is not None:
                        mapb_proto_loss.append(float(mapb_aux["prototype_loss"].detach().item()))
                    if mapb_aux.get("prototype_ready_ratio") is not None:
                        mapb_ready_ratio.append(float(mapb_aux["prototype_ready_ratio"].detach().item()))
                    if mapb_aux.get("prototype_fallback_ratio") is not None:
                        mapb_fallback_ratio.append(float(mapb_aux["prototype_fallback_ratio"].detach().item()))
            if uses_prototype_score(args):
                logger.info(
                    "Epoch: {}/{}, Loss: {:.6f}, MAPB_loss: {:.6f}, MAPB_ready_ratio: {:.6f}, MAPB_fallback_ratio: {:.6f}".format(
                        epoch,
                        args.epochs,
                        np.mean(total_loss),
                        np.mean(mapb_proto_loss) if len(mapb_proto_loss) > 0 else 0.0,
                        np.mean(mapb_ready_ratio) if len(mapb_ready_ratio) > 0 else 0.0,
                        np.mean(mapb_fallback_ratio) if len(mapb_fallback_ratio) > 0 else 0.0,
                    )
                )
            else:
                logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))
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

    parser.add_argument('--use_mvti', type=int, default=0, help='enable horizontal-flip multi-view test-time inference')

    parser.add_argument('--use_mapb', type=int, default=0, help='enable MAPB prototype-bank scoring branch')

    parser.add_argument('--score_mode', type=str, default='clip', choices=['clip', 'prototype'], help='score computation mode; prototype activates MAPB path')

    parser.add_argument('--lambda_proto', type=float, default=0.1, help='loss weight for MAPB prototype-bank objective')

    parser.add_argument('--prototype_k', type=int, default=4, help='number of MAPB prototypes per branch')

    parser.add_argument('--prototype_momentum', type=float, default=0.95, help='EMA momentum for MAPB prototype updates')

    parser.add_argument('--prototype_temperature', type=float, default=0.07, help='temperature for MAPB prototype contrastive loss')

    parser.add_argument('--prototype_max_samples', type=int, default=4096, help='maximum normal patch tokens sampled per branch for MAPB updates')

    parser.add_argument('--prototype_fusion_alpha', type=float, default=0.25, help='fusion weight between CLIP anomaly map and MAPB anomaly map')
    
    
    args = parser.parse_args()
    if args.use_mapb:
        args.score_mode = 'prototype'
    elif args.score_mode == 'prototype':
        args.use_mapb = 1
    
    args.seed = setup_seed(args.seed)
    train(args)
    
    
    
