from collections import OrderedDict
from typing import Tuple, Union
from datetime import datetime
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from clip.adaptor import Adaptor
from clip.cap_prompt import CompoundAbnormalityPrompt
from clip.prototype_bank import HypersphericalPrototypeBank


class LayerAdaptorResidual(nn.Module):
    def __init__(self, dim, bottleneck_ratio=4, zero_init=True):
        super().__init__()
        hidden = max(dim // max(int(bottleneck_ratio), 1), 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)

def gaussian_kernel(size, sigma=2.0):
    
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(x, y, indexing='ij')
    
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def tensor_shape_list(tensor):
    if tensor is None:
        return None
    return list(tensor.shape)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))
        
    def forward(self, x: torch.Tensor, fearure_layers=None, visual_prompt=None):
        out = []
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0
        for i in range(len(self.resblocks)):
            if i < prefix_len:
                x = torch.cat([visual_prompt[i:i+1].repeat(x.size(0), 1, 1), x], dim=1)
            x = self.resblocks[i](x)
            if i < prefix_len:
                x = x[:, visual_prompt[i:i+1].size(1):]
            if fearure_layers is not None and i+1 in fearure_layers:
                out.append(x)
        if fearure_layers is None:
            return x
        else:
            return out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())

    
    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(x.shape[-1], new_side * new_side).transpose(0, 1)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)
            
        x = x + self.positional_embedding.to(x.dtype)
         
        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x, feature_layers)
        for i, o in enumerate(out):
            out[i] = o.permute(1, 0, 2)
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]
        return out


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _set_prompt_token_buffer(self, name, value):
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def _fg_prompt_enabled(self):
        return getattr(self, "fg_prompt_mode", "off") == "on"

    def _cap_prompt_enabled(self):
        return bool(getattr(self, "use_cap_prompt", False)) and getattr(self, "cap_prompt", None) is not None

    def _get_prompt_trainable_parameters(self):
        if self._cap_prompt_enabled():
            return list(self.cap_prompt.parameters())
        if self._fg_prompt_enabled():
            return [self.normal_prompt_embedding, self.abnormal_prompt_embedding]
        return [self.state_prompt_embedding]

    def _get_ab_agg_mode(self, args):
        mode = getattr(args, "ab_agg", "sum_prob")
        if mode not in {"sum_prob", "max_prob", "mean_prob", "logsumexp_logit"}:
            raise ValueError("Unsupported ab_agg mode: {}".format(mode))
        return mode

    def _get_cap_agg_mode(self, args=None):
        mode = getattr(args, "cap_abnormal_agg", None) if args is not None else None
        if mode in (None, ""):
            mode = getattr(self, "cap_abnormal_agg", "mean_feature")
        if mode not in {"mean_feature", "prob_sum", "max_logit"}:
            raise ValueError("Unsupported cap_abnormal_agg mode: {}".format(mode))
        return mode

    def _aggregate_abnormal_scores(self, logits, args):
        if logits.shape[-1] < 2:
            raise ValueError("Expected at least one abnormal anchor")
        full_probs = torch.softmax(logits, dim=-1)
        abnormal_probs = full_probs[..., 1:]
        if abnormal_probs.shape[-1] == 1:
            return abnormal_probs[..., 0], abnormal_probs, full_probs

        mode = self._get_ab_agg_mode(args)
        if mode == "sum_prob":
            anomaly_score = abnormal_probs.sum(dim=-1)
        elif mode == "max_prob":
            anomaly_score = abnormal_probs.max(dim=-1).values
        elif mode == "mean_prob":
            anomaly_score = abnormal_probs.mean(dim=-1)
        elif mode == "logsumexp_logit":
            abnormal_logit = torch.logsumexp(logits[..., 1:], dim=-1)
            binary_logits = torch.stack([logits[..., 0], abnormal_logit], dim=-1)
            anomaly_score = torch.softmax(binary_logits, dim=-1)[..., 1]
        else:
            raise ValueError("Unsupported ab_agg mode: {}".format(mode))
        return anomaly_score, abnormal_probs, full_probs

    def _aggregate_cap_scores(self, logits, args):
        if logits.shape[-1] < 2:
            raise ValueError("Expected CAP logits with one normal and at least one abnormal anchor")
        full_probs = torch.softmax(logits, dim=-1)
        abnormal_probs = full_probs[..., 1:]
        mode = self._get_cap_agg_mode(args)
        if mode == "mean_feature":
            if logits.shape[-1] != 2:
                raise ValueError("CAP mean_feature expects binary text features")
            anomaly_score = abnormal_probs[..., 0]
        elif mode == "prob_sum":
            anomaly_score = abnormal_probs.sum(dim=-1)
        elif mode == "max_logit":
            abnormal_logit = logits[..., 1:].max(dim=-1).values
            binary_logits = torch.stack([logits[..., 0], abnormal_logit], dim=-1)
            anomaly_score = torch.softmax(binary_logits, dim=-1)[..., 1]
        else:
            raise ValueError("Unsupported cap_abnormal_agg mode: {}".format(mode))
        return anomaly_score, abnormal_probs, full_probs

    def _encode_prompt_embedding(self, prompt_embedding, prompt_tokens):
        if prompt_tokens is None:
            raise ValueError("Prompt tokens are not initialized")
        if prompt_embedding.ndim == 2:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        if prompt_embedding.ndim != 3:
            raise ValueError("Expected prompt embedding with shape [N, prompt_len, dim]")
        if prompt_tokens.ndim != 2:
            raise ValueError("Expected prompt tokens with shape [N, context_length]")
        if prompt_embedding.shape[0] == 1 and prompt_tokens.shape[0] > 1:
            prompt_embedding = prompt_embedding.expand(prompt_tokens.shape[0], -1, -1)
        elif prompt_tokens.shape[0] == 1 and prompt_embedding.shape[0] > 1:
            prompt_tokens = prompt_tokens.expand(prompt_embedding.shape[0], -1)
        elif prompt_embedding.shape[0] != prompt_tokens.shape[0]:
            raise ValueError(
                "Prompt embedding count {} does not match token count {}".format(
                    prompt_embedding.shape[0],
                    prompt_tokens.shape[0],
                )
            )
        state_x = self.token_embedding(prompt_tokens).type(self.dtype)
        state_x = torch.cat([prompt_embedding, state_x], dim=1)[:, :self.context_length, :]
        state_x = state_x + self.positional_embedding[:state_x.shape[1]].type(self.dtype)
        state_x = state_x.permute(1, 0, 2)
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)
        state_x = self.ln_final(state_x).type(self.dtype)
        eos_indices = prompt_embedding.shape[1] + prompt_tokens.argmax(dim=-1)
        state_x = state_x[torch.arange(state_x.shape[0], device=state_x.device), eos_indices] @ self.text_projection
        return state_x

    def _encode_cap_prompt_embedding(self, prompt_embedding, prompt_tokens):
        if prompt_tokens is None:
            raise ValueError("CAP prompt tokens are not initialized")
        if prompt_embedding.ndim == 2:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        if prompt_embedding.ndim != 3:
            raise ValueError("Expected CAP prompt embedding with shape [N, ctx_len, dim]")
        if prompt_tokens.ndim != 2:
            raise ValueError("Expected CAP prompt tokens with shape [N, context_length]")
        if prompt_embedding.shape[0] == 1 and prompt_tokens.shape[0] > 1:
            prompt_embedding = prompt_embedding.expand(prompt_tokens.shape[0], -1, -1)
        elif prompt_tokens.shape[0] == 1 and prompt_embedding.shape[0] > 1:
            prompt_tokens = prompt_tokens.expand(prompt_embedding.shape[0], -1)
        elif prompt_embedding.shape[0] != prompt_tokens.shape[0]:
            raise ValueError(
                "CAP prompt embedding count {} does not match token count {}".format(
                    prompt_embedding.shape[0],
                    prompt_tokens.shape[0],
                )
            )
        token_x = self.token_embedding(prompt_tokens).type(self.dtype)
        # CAP uses the FAPrompt-style layout:
        # [SOS] + learnable context tokens + fixed semantic anchor tokens + [EOT]
        state_x = torch.cat([token_x[:, :1, :], prompt_embedding, token_x[:, 1:, :]], dim=1)[:, :self.context_length, :]
        state_x = state_x + self.positional_embedding[:state_x.shape[1]].type(self.dtype)
        state_x = state_x.permute(1, 0, 2)
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)
        state_x = self.ln_final(state_x).type(self.dtype)
        eos_indices = prompt_embedding.shape[1] + prompt_tokens.argmax(dim=-1)
        state_x = state_x[torch.arange(state_x.shape[0], device=state_x.device), eos_indices] @ self.text_projection
        return state_x

    def _build_cap_prompt_outputs(self, args=None):
        if not self._cap_prompt_enabled():
            return None
        return self.cap_prompt(
            encode_fn=self._encode_cap_prompt_embedding,
            abnormal_agg=self._get_cap_agg_mode(args),
        )

    def insert(self, args, tokenizer, device):
        self.normal_cls_prompt = f'without defect.'
        self.anomaly_cls_prompt = f'with defect.'
        self.use_cap_prompt = bool(getattr(args, "use_cap_prompt", False))
        self.cap_abnormal_agg = getattr(args, "cap_abnormal_agg", "mean_feature")
        self.cap_log_interval = max(int(getattr(args, "cap_log_interval", 1)), 1)
        self.fg_prompt_mode = getattr(args, "fg_prompt", "off")
        self.num_ab_prompts = max(int(getattr(args, "num_ab_prompts", 4)), 1)
        if self.fg_prompt_mode not in {"off", "on"}:
            raise ValueError("Unsupported fg_prompt mode: {}".format(self.fg_prompt_mode))
        if self._fg_prompt_enabled():
            self._set_prompt_token_buffer("state_prompt_tokens", None)
            self._set_prompt_token_buffer(
                "normal_prompt_tokens",
                tokenizer([self.normal_cls_prompt]).to(device),
            )
            self._set_prompt_token_buffer(
                "abnormal_prompt_tokens",
                tokenizer([self.anomaly_cls_prompt]).to(device),
            )
        else:
            self._set_prompt_token_buffer(
                "state_prompt_tokens",
                tokenizer([self.normal_cls_prompt, self.anomaly_cls_prompt]).to(device),
            )
            self._set_prompt_token_buffer("normal_prompt_tokens", None)
            self._set_prompt_token_buffer("abnormal_prompt_tokens", None)
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_len = args.prompt_len
        prompt_dim = self.token_embedding.weight.shape[-1]
        if self._fg_prompt_enabled():
            normal_prompt_seed = torch.empty(1, args.prompt_len, prompt_dim, device=device)
            abnormal_prompt_seed = torch.empty(self.num_ab_prompts, args.prompt_len, prompt_dim, device=device)
            nn.init.normal_(normal_prompt_seed, std=0.01)
            nn.init.normal_(abnormal_prompt_seed, std=0.01)
            self.state_prompt_embedding = None
            self.normal_prompt_embedding = nn.Parameter(normal_prompt_seed)
            self.abnormal_prompt_embedding = nn.Parameter(abnormal_prompt_seed)
            self.normal_prompt_embedding.requires_grad_(True)
            self.abnormal_prompt_embedding.requires_grad_(True)
        else:
            prompt_seed = torch.empty(1, args.prompt_len, prompt_dim, device=device)
            nn.init.normal_(prompt_seed, std=0.01)
            self.state_prompt_embedding = nn.Parameter(prompt_seed)
            self.state_prompt_embedding.requires_grad_(True)
            self.normal_prompt_embedding = None
            self.abnormal_prompt_embedding = None
        self.cap_prompt = None
        if self.use_cap_prompt:
            self.cap_prompt = CompoundAbnormalityPrompt(
                tokenizer=tokenizer,
                normal_prompt_text=self.normal_cls_prompt,
                abnormal_prompt_text=self.anomaly_cls_prompt,
                text_width=prompt_dim,
                num_abnormal_prompts=getattr(args, "cap_num_abnormal_prompts", 10),
                n_normal_ctx=getattr(args, "cap_n_normal_ctx", 4),
                n_abnormal_ctx=getattr(args, "cap_n_abnormal_ctx", 4),
                ctx_init=getattr(args, "cap_ctx_init", "random"),
                device=device,
            ).to(device)
            if self.state_prompt_embedding is not None:
                self.state_prompt_embedding.requires_grad_(False)
            if self.normal_prompt_embedding is not None:
                self.normal_prompt_embedding.requires_grad_(False)
            if self.abnormal_prompt_embedding is not None:
                self.abnormal_prompt_embedding.requires_grad_(False)
        self.tokenizer = tokenizer
        
        self.adaptor =  Adaptor(inplanes=self.visual.proj.shape[0], outplanes=self.visual.proj.shape[0]).to(device)
        self.layer_residuals = None
        if bool(getattr(args, "use_lsar", 0)):
            num_feature_layers = len(getattr(args, "feature_layers", []))
            self.layer_residuals = nn.ModuleList(
                [
                    LayerAdaptorResidual(
                        dim=self.visual.proj.shape[1],
                        bottleneck_ratio=getattr(args, "lsar_bottleneck_ratio", 4),
                        zero_init=bool(getattr(args, "lsar_zero_init", 1)),
                    )
                    for _ in range(num_feature_layers)
                ]
            ).to(device)
        self.score_mode = getattr(args, "score_mode", "clip")
        self.prototype_bank = None
        if self._fg_prompt_enabled():
            self.mapb_requested_branch_num = self.num_ab_prompts
            self.mapb_default_branch_num = self.num_ab_prompts
            self.mapb_effective_branch_num = self.num_ab_prompts
        else:
            self.mapb_requested_branch_num = 0
            self.mapb_default_branch_num = 0
            self.mapb_effective_branch_num = 0
        if self.score_mode == "prototype":
            num_feature_layers = max(len(getattr(args, "feature_layers", [])), 1)
            branches_per_layer = 3
            self.prototype_bank = HypersphericalPrototypeBank(
                num_branches=max(num_feature_layers * branches_per_layer, 1),
                num_prototypes=getattr(args, "prototype_k", 4),
                dim=self.visual.proj.shape[1],
                momentum=getattr(args, "prototype_momentum", 0.95),
            ).to(device)
        self._mapb_debug_dumped = False
        self._mapb_debug_first_forward = None
        self._mapb_debug_model_config = self.build_mapb_debug_config(args)
        self._latest_mapb_aux = None
        self._latest_cap_aux = None
        self._latest_prompt_debug = None
        self._latest_prompt_analysis = None
        self.memorybank = None
        self.memory_backbone = None
        self.gaussian_kernel = {'3': gaussian_kernel(size=3, sigma=4).to(device), '5': gaussian_kernel(size=5, sigma=4).to(device)}
        self.emit_mapb_debug_logs(args)

    def build_mapb_debug_config(self, args):
        if self._cap_prompt_enabled():
            abnormal_prompt_count = int(self.cap_prompt.num_abnormal_prompts)
            normal_prompt_tokens = self.cap_prompt.normal_prompt_tokens
            abnormal_prompt_tokens = self.cap_prompt.abnormal_prompt_tokens
            learnable_context_shape = tensor_shape_list(self.cap_prompt.abnormal_ctx)
            normal_prompt_embedding_shape = tensor_shape_list(self.cap_prompt.normal_ctx.unsqueeze(0))
            abnormal_prompt_embedding_shape = tensor_shape_list(self.cap_prompt.build_prompt_embeddings()[1])
            prompt_note = "CAP is enabled as a text-side compound abnormality prompt module, independent from MAPB."
        else:
            abnormal_prompt_count = self.num_ab_prompts if self._fg_prompt_enabled() else 1
            normal_prompt_tokens = self.normal_prompt_tokens if self._fg_prompt_enabled() else self.state_prompt_tokens[:1]
            abnormal_prompt_tokens = self.abnormal_prompt_tokens if self._fg_prompt_enabled() else self.state_prompt_tokens[1:2]
            learnable_context_shape = tensor_shape_list(self.abnormal_prompt_embedding if self._fg_prompt_enabled() else self.state_prompt_embedding)
            normal_prompt_embedding_shape = tensor_shape_list(self.normal_prompt_embedding if self._fg_prompt_enabled() else self.state_prompt_embedding)
            abnormal_prompt_embedding_shape = tensor_shape_list(self.abnormal_prompt_embedding if self._fg_prompt_enabled() else self.state_prompt_embedding)
            prompt_note = "Legacy text-prompt path is active."
        prompt_config = {
            "normal_prompt": self.normal_cls_prompt,
            "abnormal_prompts": [self.anomaly_cls_prompt] * abnormal_prompt_count,
            "normal_prompt_token_shape": tensor_shape_list(normal_prompt_tokens),
            "abnormal_prompt_token_shape": tensor_shape_list(abnormal_prompt_tokens),
            "learnable_context_shape": learnable_context_shape,
            "normal_prompt_embedding_shape": normal_prompt_embedding_shape,
            "abnormal_prompt_embedding_shape": abnormal_prompt_embedding_shape,
            "text_prompt_count": int(1 + abnormal_prompt_count),
            "abnormal_prompt_count": int(abnormal_prompt_count),
            "text_prototype_order": ["normal"] + [f"abnormal_{idx + 1}" for idx in range(abnormal_prompt_count)],
            "note": prompt_note,
        }
        model_config = {
            "mapb_module_exists": self.prototype_bank is not None,
            "mapb_class_name": self.prototype_bank.__class__.__name__ if self.prototype_bank is not None else None,
            "use_mapb": int(self.prototype_bank is not None),
            "requested_branch_num": int(getattr(args, "mapb_branch_num", 0)),
            "legacy_branch_count_arg": int(getattr(args, "mapb_branch_count", 0) or 0),
            "default_branch_num": int(getattr(self.prototype_bank, "num_branches", 0) or 0),
            "effective_branch_num": int(getattr(self.prototype_bank, "num_branches", 0) or 0),
            "abnormal_branch_count_semantic": "visual prototype-bank branches over patch-token feature groups",
            "normal_branch_count": 1,
            "aggregation_type": self._get_ab_agg_mode(args),
            "score_mode": getattr(args, "score_mode", "clip"),
            "prototype_k": int(getattr(args, "prototype_k", 4)),
            "prototype_fusion_alpha": float(getattr(args, "prototype_fusion_alpha", 0.25)),
            "prototype_bank_shape": tensor_shape_list(self.prototype_bank.prototypes) if self.prototype_bank is not None else None,
            "prototype_initialized_shape": tensor_shape_list(self.prototype_bank.initialized) if self.prototype_bank is not None else None,
            "feature_layers": list(getattr(args, "feature_layers", [])),
            "memory_layers": list(getattr(args, "memory_layers", [])),
            "branches_per_layer": 3,
        }
        return {
            "model_mapb_config": model_config,
            "prompt_config": prompt_config,
            "text_feature_shapes": {
                "state_prompt_tokens": tensor_shape_list(self.state_prompt_tokens),
                "normal_prompt_tokens": tensor_shape_list(self.normal_prompt_tokens),
                "abnormal_prompt_tokens": tensor_shape_list(self.abnormal_prompt_tokens),
            },
            "aggregation_config": {
                "aggregation": self._get_ab_agg_mode(args),
                "score_mode": getattr(args, "score_mode", "clip"),
            },
        }

    def emit_mapb_debug_logs(self, args):
        if not bool(getattr(args, "debug_mapb", 0)):
            return
        config = self._mapb_debug_model_config["model_mapb_config"]
        prompt_config = self._mapb_debug_model_config["prompt_config"]
        print(f"[MAPB CONFIG] use_mapb={config['use_mapb']}")
        print(f"[MAPB CONFIG] requested_branch_num={config['requested_branch_num']}")
        print(f"[MAPB CONFIG] default_branch_num={config['default_branch_num']}")
        print(f"[MAPB CONFIG] effective_branch_num={config['effective_branch_num']}")
        print(f"[MAPB CONFIG] abnormal_prompt_count={prompt_config['abnormal_prompt_count']}")
        print(f"[MAPB CONFIG] text_prompt_count={prompt_config['text_prompt_count']}")
        print(f"[MAPB CONFIG] text_prototype_order={prompt_config['text_prototype_order']}")
        print(f"[MAPB CONFIG] learnable_context_shape={prompt_config['learnable_context_shape']}")
        print(f"[MAPB CONFIG] normal_prompt_token_shape={prompt_config['normal_prompt_token_shape']}")
        print(f"[MAPB CONFIG] abnormal_prompt_token_shape={prompt_config['abnormal_prompt_token_shape']}")
        print(f"[MAPB CONFIG] prototype_bank_shape={config['prototype_bank_shape']}")
        print(f"[MAPB CONFIG] aggregation={config['aggregation_type']}")
        print(f"[MAPB CONFIG] score_mode={config['score_mode']}")
        if config["requested_branch_num"] <= 0:
            print(f"[MAPB CONFIG] mapb_branch_num=0 means using default={config['default_branch_num']}")

    def record_mapb_first_forward(self, args, image, img_tokens, text_features, branch_logit_shapes, scores, predict_map, cls_label):
        if not bool(getattr(args, "debug_mapb", 0)) or self._mapb_debug_first_forward is not None:
            return
        first_forward = {
            "image_batch_shape": tensor_shape_list(image),
            "selected_feature_layers": list(getattr(args, "feature_layers", [])),
            "patch_feature_shapes_per_branch": [tensor_shape_list(token[:, 1:, :]) for token in img_tokens],
            "class_feature_shapes_per_branch": [tensor_shape_list(token[:, :1, :]) for token in img_tokens],
            "text_feature_shape": tensor_shape_list(text_features),
            "branch_logit_shapes": branch_logit_shapes,
            "normal_logits_shape": tensor_shape_list(scores[..., 0]),
            "abnormal_logits_shape": tensor_shape_list(scores[..., 1:]),
            "final_abnormal_probability_shape": tensor_shape_list(predict_map),
            "image_level_score_shape": tensor_shape_list(cls_label),
        }
        if self._latest_mapb_aux is not None:
            for key in ["prototype_ready_ratio", "prototype_fallback_ratio", "prototype_loss", "prototype_loss_branches"]:
                value = self._latest_mapb_aux.get(key)
                if value is not None:
                    first_forward[key] = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        self._mapb_debug_first_forward = first_forward
        print(f"[MAPB FORWARD] image_batch_shape={first_forward['image_batch_shape']}")
        print(f"[MAPB FORWARD] selected_feature_layers={first_forward['selected_feature_layers']}")
        print(f"[MAPB FORWARD] patch_feature_shapes_per_branch={first_forward['patch_feature_shapes_per_branch']}")
        print(f"[MAPB FORWARD] class_feature_shapes_per_branch={first_forward['class_feature_shapes_per_branch']}")
        print(f"[MAPB FORWARD] text_feature_shape={first_forward['text_feature_shape']}")
        print(f"[MAPB FORWARD] branch_logit_shapes={first_forward['branch_logit_shapes']}")
        print(f"[MAPB FORWARD] abnormal_logits_shape={first_forward['abnormal_logits_shape']}")
        print(f"[MAPB FORWARD] final_abnormal_probability_shape={first_forward['final_abnormal_probability_shape']}")
        print(f"[MAPB FORWARD] image_level_score_shape={first_forward['image_level_score_shape']}")
        if "prototype_ready_ratio" in first_forward:
            print(f"[MAPB FORWARD] ready_ratio={first_forward['prototype_ready_ratio']}")
        if "prototype_fallback_ratio" in first_forward:
            print(f"[MAPB FORWARD] fallback_ratio={first_forward['prototype_fallback_ratio']}")
        self.maybe_dump_mapb_debug_json(args)

    def maybe_dump_mapb_debug_json(self, args):
        if self._mapb_debug_dumped or not bool(getattr(args, "debug_mapb", 0)) or self._mapb_debug_first_forward is None:
            return
        json_path = getattr(args, "mapb_debug_json", None)
        if not json_path:
            return
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        payload = {
            "experiment_name": os.path.basename(os.path.abspath(getattr(args, "log_dir", "mapb_debug"))),
            "full_command": getattr(args, "_command", ""),
            "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
            "model_mapb_config": self._mapb_debug_model_config["model_mapb_config"],
            "prompt_config": self._mapb_debug_model_config["prompt_config"],
            "text_feature_shapes": {
                **self._mapb_debug_model_config["text_feature_shapes"],
                "text_features": self._mapb_debug_first_forward.get("text_feature_shape"),
                "final_text_prototype_shape": self._mapb_debug_first_forward.get("text_feature_shape"),
                "normal_prototype_shape": [1, self._mapb_debug_first_forward["text_feature_shape"][-1]] if self._mapb_debug_first_forward.get("text_feature_shape") else None,
                "abnormal_prototype_shape": [max(self._mapb_debug_first_forward["text_feature_shape"][0] - 1, 0), self._mapb_debug_first_forward["text_feature_shape"][-1]] if self._mapb_debug_first_forward.get("text_feature_shape") else None,
            },
            "aggregation_config": self._mapb_debug_model_config["aggregation_config"],
            "first_forward_shapes": self._mapb_debug_first_forward,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self._mapb_debug_dumped = True
        
    
    def encode_state_prompt(self, args=None):
        if self._cap_prompt_enabled():
            return self._build_cap_prompt_outputs(args)["text_features_for_scoring"]
        if self._fg_prompt_enabled():
            normal_text_features = self._encode_prompt_embedding(
                self.normal_prompt_embedding,
                self.normal_prompt_tokens,
            )
            abnormal_text_features = self._encode_prompt_embedding(
                self.abnormal_prompt_embedding,
                self.abnormal_prompt_tokens,
            )
            return torch.cat([normal_text_features, abnormal_text_features], dim=0)
        return self._encode_prompt_embedding(self.state_prompt_embedding, self.state_prompt_tokens)
    
    
    def get_trainable_parameters(self):
        params = list(self._get_prompt_trainable_parameters()) + list(self.adaptor.parameters())
        if self.layer_residuals is not None:
            params += list(self.layer_residuals.parameters())
        return params

    def use_prototype_score(self, args):
        score_mode = getattr(args, "score_mode", self.score_mode)
        return score_mode == "prototype" and self.prototype_bank is not None

    def build_patch_normal_mask(self, gts, spatial_size):
        if gts is None:
            return None
        resized_gts = F.interpolate(gts.float(), size=spatial_size, mode="bilinear", align_corners=False)
        return (resized_gts <= 0.5).squeeze(1)

    def update_prototype_bank(self, img_tokens, normal_mask, args):
        if normal_mask is None or not self.use_prototype_score(args):
            return None
        proto_aux = self.prototype_bank.update(
            img_tokens,
            normal_mask,
            max_samples=getattr(args, "prototype_max_samples", 0),
        )
        proto_loss, loss_aux = self.prototype_bank.loss(
            img_tokens,
            normal_mask,
            temperature=getattr(args, "prototype_temperature", 0.07),
            max_samples=getattr(args, "prototype_max_samples", 0),
        )
        proto_aux = dict(proto_aux)
        proto_aux.update(loss_aux)
        proto_aux["prototype_loss_value"] = proto_loss
        return proto_aux

    def compute_prototype_branch_maps(self, img_tokens, fallback_maps):
        if self.prototype_bank is None:
            return fallback_maps, None

        branch_maps = []
        branch_distance_means = []
        fallback_count = 0
        for branch_idx, (img_token, fallback_map) in enumerate(zip(img_tokens, fallback_maps)):
            distances = self.prototype_bank.branch_distances(branch_idx, img_token)
            if distances is None:
                branch_map = fallback_map
                fallback_count += 1
            else:
                b, l = distances.shape
                h = w = int(math.sqrt(l))
                branch_map = distances.reshape(b, h, w)
                branch_distance_means.append(branch_map.mean())
            branch_maps.append(branch_map)

        aux = {
            "prototype_ready_ratio": self.prototype_bank.initialized_ratio(),
            "prototype_fallback_ratio": fallback_maps[0].new_tensor(float(fallback_count) / max(len(fallback_maps), 1)),
            "prototype_distance_mean": torch.stack(branch_distance_means).mean()
            if len(branch_distance_means) > 0
            else fallback_maps[0].new_tensor(0.0),
        }
        return branch_maps, aux

    def build_branch_average_map(self, branch_maps, aggregation="mean"):
        stacked_maps = torch.stack(branch_maps, dim=1)
        if aggregation == "max":
            return stacked_maps.max(dim=1, keepdim=True)[0]
        if aggregation == "logsumexp":
            return torch.logsumexp(stacked_maps, dim=1, keepdim=True) - math.log(max(stacked_maps.shape[1], 1))
        return stacked_maps.mean(dim=1, keepdim=True)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image, feature_layers=None):
        return self.visual(image.type(self.dtype), feature_layers)
    
    def aggerate_neighbor(self, x, patchsize, stride=1):
        if patchsize == 1:
            return x
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        padding = patchsize // 2
        b, l, c = x.size()
        h = w = int(math.sqrt(l))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        x = torch.nn.functional.unfold(x, kernel_size=patchsize, padding=padding, stride=stride) #b, (c * r * r), h * w
        x = x.permute(0, 2, 1).reshape(-1, c, patchsize * patchsize).permute(0, 2, 1) # (b * h * w,  r * r, c)
        kernel = self.gaussian_kernel[str(patchsize)].reshape(1, patchsize * patchsize, 1)
        x = torch.sum(x * kernel, dim=1).reshape(b, l, c)
        x = torch.cat([cls_token, x], dim=1)
        return x
    
    
    def aggerate_neighbors(self, img_tokens, args=None):
        img_token_list = []
        for img_token in img_tokens:
            for r in [1, 3, 5]:
                new_img_token = self.aggerate_neighbor(img_token, r)
                img_token_list.append(new_img_token)
        return img_token_list
    
    
    def detect_encode_image(self, image, args):
        img_tokens = self.encode_image(image, args.feature_layers) 
        img_tokens = self.aggerate_neighbors(img_tokens, args=args)
        projected_tokens = [self.visual.ln_post(self.adaptor(img_token)) @ self.visual.proj for img_token in img_tokens]
        if self.layer_residuals is None:
            return projected_tokens
        branches_per_layer = 3
        refined_tokens = []
        for branch_idx, img_token in enumerate(projected_tokens):
            layer_idx = min(branch_idx // branches_per_layer, len(self.layer_residuals) - 1)
            refined_tokens.append(self.layer_residuals[layer_idx](img_token))
        img_tokens = refined_tokens
        return img_tokens
    
    
    def store_memory(self, image, args):
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens, args=args)
        b, l, c = img_tokens[0].size()
        self.memorybank = [torch.nn.functional.normalize(img_token[:, 1:], dim=-1).reshape(-1, c) for img_token in img_tokens]
        
    
    def detect_forward_seg(self, image, args, gts=None):
        cap_outputs = self._build_cap_prompt_outputs(args)
        text_features = cap_outputs["text_features_for_scoring"] if cap_outputs is not None else self.encode_state_prompt(args=args)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        img_tokens = self.detect_encode_image(image, args)
        scores = None
        clip_branch_maps = []
        branch_logit_shapes = []
        for img_token in img_tokens:
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = torch.matmul(img_token, text_features.permute(1, 0)) / 0.07
            branch_logit_shapes.append(tensor_shape_list(score))
            scores = score if scores is None else scores + score
            if cap_outputs is not None:
                branch_anomaly_score, _, _ = self._aggregate_cap_scores(score, args)
            else:
                branch_anomaly_score, _, _ = self._aggregate_abnormal_scores(score, args)
            branch_predict_map = branch_anomaly_score[:, 1:]
            b, l = branch_predict_map.size()
            h = w = int(math.sqrt(l))
            clip_branch_maps.append(branch_predict_map.reshape(b, h, w))
        if cap_outputs is not None:
            anomaly_score, abnormal_prob, full_prob = self._aggregate_cap_scores(scores, args)
        else:
            anomaly_score, abnormal_prob, full_prob = self._aggregate_abnormal_scores(scores, args)
        cls_label = anomaly_score[:, 0].view(-1)
        predict_map = anomaly_score[:, 1:]
        self._latest_cap_aux = None
        if cap_outputs is not None:
            self._latest_cap_aux = {
                **cap_outputs["diagnostics"],
                "cap_orth_loss": cap_outputs["cap_orth_loss"],
            }
        self._latest_prompt_debug = {
            "prompt_source": "cap" if cap_outputs is not None else ("legacy_fg" if self._fg_prompt_enabled() else "legacy_state"),
            "fg_prompt": self.fg_prompt_mode,
            "use_cap_prompt": self._cap_prompt_enabled(),
            "num_ab_prompts": int(cap_outputs["diagnostics"]["cap_num_abnormal_prompts"]) if cap_outputs is not None else int(self.num_ab_prompts),
            "ab_agg": self._get_ab_agg_mode(args),
            "cap_abnormal_agg": self._get_cap_agg_mode(args) if cap_outputs is not None else None,
            "prototype_score_active": bool(self.use_prototype_score(args)),
            "text_feature_shape": list(text_features.shape),
            "scale_text_feature_shape": None,
            "image_cls_logits_shape": list(scores[:, :1, :].shape),
            "image_cls_prob_shape": list(full_prob[:, :1, :].shape),
            "pixel_score_shape": list(scores[:, 1:, :].shape),
            "pixel_prob_shape": list(full_prob[:, 1:, :].shape),
        }
        if (self._fg_prompt_enabled() or self._cap_prompt_enabled()) and getattr(args, "dump_prompt_diag_json", ""):
            self._latest_prompt_analysis = {
                "ab_agg": self._get_cap_agg_mode(args) if cap_outputs is not None else self._get_ab_agg_mode(args),
                "abnormal_text_features": (
                    cap_outputs["abnormal_text_features"].detach().float().cpu()
                    if cap_outputs is not None
                    else text_features[1:].detach().float().cpu()
                ),
                "image_abnormal_probs": abnormal_prob[:, 0].detach().float().cpu(),
                "pixel_abnormal_probs": abnormal_prob[:, 1:].detach().float().cpu(),
            }
        else:
            self._latest_prompt_analysis = None
        
        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        clip_base_map = predict_map.reshape(b, 1, h, w)
        self._latest_mapb_aux = None
        if self.use_prototype_score(args):
            normal_mask = self.build_patch_normal_mask(gts, spatial_size=(h, w))
            score_aux = self.update_prototype_bank(img_tokens, normal_mask, args)
            proto_branch_maps, proto_aux = self.compute_prototype_branch_maps(img_tokens, clip_branch_maps)
            if proto_aux is not None:
                if score_aux is None:
                    score_aux = {}
                score_aux.update(proto_aux)
            fusion_alpha = float(getattr(args, "prototype_fusion_alpha", 0.25))
            fusion_alpha = min(max(fusion_alpha, 0.0), 1.0)
            if proto_aux is not None and float(proto_aux["prototype_fallback_ratio"].detach().item()) >= 1.0:
                predict_map = clip_base_map
            else:
                proto_base_map = self.build_branch_average_map(
                    proto_branch_maps,
                    aggregation=getattr(args, "mapb_aggregation", "mean"),
                )
                predict_map = torch.lerp(clip_base_map, proto_base_map, fusion_alpha)
                if score_aux is None:
                    score_aux = {}
                score_aux["prototype_fusion_alpha"] = clip_base_map.new_tensor(fusion_alpha)
                score_aux["prototype_map_mean"] = proto_base_map.mean()
                score_aux["prototype_map_delta"] = (proto_base_map - clip_base_map).abs().mean()
            self._latest_mapb_aux = score_aux
            self._latest_prompt_debug["prototype_fusion_applied"] = True
        else:
            predict_map = clip_base_map
            self._latest_prompt_debug["prototype_fusion_applied"] = False
        self.record_mapb_first_forward(args, image, img_tokens, text_features, branch_logit_shapes, scores, predict_map, cls_label)
        return cls_label, predict_map, img_tokens
        
    
    def detect_forward_memorybank(self, image, args):
        scores = 0
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        for i, img_token in enumerate(img_tokens):
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = (1 - torch.matmul(img_token, self.memorybank[i].T)) .min(dim=-1)[0] / 2
            scores += score[:, 1:]
        scores = scores / len(img_tokens)
        cls_label = torch.max(scores, dim=-1)[0]
        b, l = scores.size()
        h = w = int(math.sqrt(l))
        predict_map = scores.reshape(b, 1, h, w)
        return cls_label, predict_map
    
    
    
    def detect_forward(self, image, args):
        cls_label, predict_map, _= self.detect_forward_seg(image, args)
        if self.memorybank is not None:
            cls_label_memory, predict_map_memory = self.detect_forward_memorybank(image, args)
            predict_map = predict_map_memory + args.alpha * predict_map
            cls_label = cls_label_memory + args.alpha * cls_label
        return cls_label, predict_map
    

    def forward(self, image, text):
        image_features = self.encode_image(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[0]
        text_features = self.encode_text(text)
        if isinstance(text_features, (list, tuple)):
            text_features = text_features[0]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
