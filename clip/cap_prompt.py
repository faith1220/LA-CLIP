import warnings

import torch
import torch.nn.functional as F
from torch import nn


class CompoundAbnormalityPrompt(nn.Module):
    """
    Text-side CAP module.

    This module is intentionally independent from MAPB. It learns:
    - one shared normal context token sequence
    - K abnormal-specific context token sequences

    During scoring, the CLIP text encoder is still reused through the provided
    encode function so the prompt features remain fully compatible with the
    repository's existing text pathway.
    """

    def __init__(
        self,
        tokenizer,
        normal_prompt_text,
        abnormal_prompt_text,
        text_width,
        num_abnormal_prompts=10,
        n_normal_ctx=4,
        n_abnormal_ctx=4,
        ctx_init="random",
        device=None,
    ):
        super().__init__()
        self.num_abnormal_prompts = max(int(num_abnormal_prompts), 1)
        self.n_normal_ctx = max(int(n_normal_ctx), 0)
        self.n_abnormal_ctx = max(int(n_abnormal_ctx), 0)
        self.text_width = int(text_width)
        self.ctx_init_requested = str(ctx_init)
        self.ctx_init_effective = "random"
        self.normal_prompt_text = str(normal_prompt_text)
        self.abnormal_prompt_text = str(abnormal_prompt_text)

        if self.ctx_init_requested != "random":
            warnings.warn(
                "CAP ctx_init='{}' is not implemented in this repository layout yet; "
                "falling back to random initialization.".format(self.ctx_init_requested)
            )

        normal_tokens = tokenizer([self.normal_prompt_text])
        abnormal_tokens = tokenizer([self.abnormal_prompt_text])
        if device is not None:
            normal_tokens = normal_tokens.to(device)
            abnormal_tokens = abnormal_tokens.to(device)
        self.register_buffer("normal_prompt_tokens", normal_tokens, persistent=False)
        self.register_buffer("abnormal_prompt_tokens", abnormal_tokens, persistent=False)

        self.normal_ctx = nn.Parameter(torch.empty(self.n_normal_ctx, self.text_width))
        self.abnormal_ctx = nn.Parameter(
            torch.empty(self.num_abnormal_prompts, self.n_abnormal_ctx, self.text_width)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.normal_ctx, std=0.01)
        nn.init.normal_(self.abnormal_ctx, std=0.01)

    def build_prompt_embeddings(self):
        normal_prompt_embedding = self.normal_ctx.unsqueeze(0)
        shared_normal = self.normal_ctx.unsqueeze(0).expand(self.num_abnormal_prompts, -1, -1)
        abnormal_prompt_embedding = torch.cat([shared_normal, self.abnormal_ctx], dim=1)
        return normal_prompt_embedding, abnormal_prompt_embedding

    def compute_orthogonal_constraint(self, abnormal_text_features):
        abnormal_norm = F.normalize(abnormal_text_features, dim=-1)
        if abnormal_norm.shape[0] <= 1:
            zero = abnormal_norm.new_tensor(0.0)
            return zero, {
                "cap_abn_pair_cos_mean": 0.0,
                "cap_abn_pair_cos_max": 0.0,
                "cap_abn_pair_cos_min": 0.0,
            }

        gram = torch.matmul(abnormal_norm, abnormal_norm.t())
        offdiag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
        offdiag_values = gram[offdiag_mask]
        orth_loss = offdiag_values.square().mean()
        stats = {
            "cap_abn_pair_cos_mean": float(offdiag_values.mean().detach().item()),
            "cap_abn_pair_cos_max": float(offdiag_values.max().detach().item()),
            "cap_abn_pair_cos_min": float(offdiag_values.min().detach().item()),
        }
        return orth_loss, stats

    def forward(self, encode_fn, abnormal_agg="mean_feature"):
        normal_prompt_embedding, abnormal_prompt_embedding = self.build_prompt_embeddings()
        normal_text_feature = encode_fn(normal_prompt_embedding, self.normal_prompt_tokens)
        abnormal_text_features = encode_fn(abnormal_prompt_embedding, self.abnormal_prompt_tokens)

        normal_text_feature = F.normalize(normal_text_feature, dim=-1)
        abnormal_text_features = F.normalize(abnormal_text_features, dim=-1)
        abnormal_text_prototype = F.normalize(
            abnormal_text_features.mean(dim=0, keepdim=True),
            dim=-1,
        )

        if abnormal_agg == "mean_feature":
            text_features_for_scoring = torch.cat(
                [normal_text_feature, abnormal_text_prototype],
                dim=0,
            )
        elif abnormal_agg in {"prob_sum", "max_logit"}:
            text_features_for_scoring = torch.cat(
                [normal_text_feature, abnormal_text_features],
                dim=0,
            )
        else:
            raise ValueError("Unsupported CAP abnormal aggregation: {}".format(abnormal_agg))

        cap_orth_loss, orth_stats = self.compute_orthogonal_constraint(abnormal_text_features)
        diagnostics = {
            "cap_orth_loss_value": float(cap_orth_loss.detach().item()),
            "cap_num_abnormal_prompts": int(self.num_abnormal_prompts),
            "cap_n_normal_ctx": int(self.n_normal_ctx),
            "cap_n_abnormal_ctx": int(self.n_abnormal_ctx),
            "cap_ctx_init_requested": self.ctx_init_requested,
            "cap_ctx_init_effective": self.ctx_init_effective,
            "cap_abnormal_agg": abnormal_agg,
            "normal_ctx_shape": list(self.normal_ctx.shape),
            "abnormal_ctx_shape": list(self.abnormal_ctx.shape),
        }
        diagnostics.update(orth_stats)

        return {
            "normal_text_feature": normal_text_feature,
            "abnormal_text_features": abnormal_text_features,
            "abnormal_text_prototype": abnormal_text_prototype,
            "text_features_for_scoring": text_features_for_scoring,
            "cap_orth_loss": cap_orth_loss,
            "diagnostics": diagnostics,
        }
