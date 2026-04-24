import math

import torch
import torch.nn.functional as F
from torch import nn


class HypersphericalPrototypeBank(nn.Module):
    def __init__(self, num_branches, num_prototypes, dim, momentum=0.95, eps=1e-6):
        super().__init__()
        self.num_branches = int(num_branches)
        self.num_prototypes = int(num_prototypes)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer("prototypes", torch.zeros(self.num_branches, self.num_prototypes, self.dim))
        self.register_buffer("initialized", torch.zeros(self.num_branches, self.num_prototypes, dtype=torch.bool))

    def initialized_ratio(self):
        return self.initialized.float().mean()

    def _subsample(self, tokens, max_samples):
        if max_samples <= 0 or tokens.shape[0] <= max_samples:
            return tokens
        perm = torch.randperm(tokens.shape[0], device=tokens.device)[:max_samples]
        return tokens[perm]

    def _initialize_branch(self, branch_idx, tokens):
        tokens = F.normalize(tokens, dim=-1)
        if tokens.shape[0] < 1:
            return
        if tokens.shape[0] >= self.num_prototypes:
            perm = torch.randperm(tokens.shape[0], device=tokens.device)[:self.num_prototypes]
            init_tokens = tokens[perm]
        else:
            repeat_count = math.ceil(self.num_prototypes / tokens.shape[0])
            init_tokens = tokens.repeat(repeat_count, 1)[: self.num_prototypes]
        self.prototypes[branch_idx].copy_(init_tokens)
        self.initialized[branch_idx].fill_(True)

    @torch.no_grad()
    def update(self, img_tokens, normal_mask, max_samples=0):
        if len(img_tokens) != self.num_branches:
            raise ValueError("Prototype bank branch count does not match img_tokens")

        flat_mask = normal_mask.reshape(normal_mask.shape[0], -1).bool()
        updated_branches = 0
        total_normal_tokens = 0

        for branch_idx, img_token in enumerate(img_tokens):
            patch_tokens = F.normalize(img_token[:, 1:, :].detach(), dim=-1)
            branch_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1])
            branch_mask = flat_mask.reshape(-1)
            branch_tokens = branch_tokens[branch_mask]
            if branch_tokens.shape[0] < 1:
                continue

            branch_tokens = self._subsample(branch_tokens, max_samples=max_samples)
            total_normal_tokens += int(branch_tokens.shape[0])

            if not self.initialized[branch_idx].all():
                self._initialize_branch(branch_idx, branch_tokens)

            prototypes = self.prototypes[branch_idx]
            sims = torch.matmul(branch_tokens, prototypes.t())
            assign_idx = sims.argmax(dim=-1)
            one_hot = F.one_hot(assign_idx, num_classes=self.num_prototypes).to(branch_tokens.dtype)
            counts = one_hot.sum(dim=0)
            if (counts > 0).any():
                means = torch.matmul(one_hot.t(), branch_tokens)
                means = means / counts.clamp_min(1.0).unsqueeze(-1)
                means = F.normalize(means, dim=-1)
                update_mask = counts > 0
                updated = prototypes[update_mask] * self.momentum + means[update_mask] * (1.0 - self.momentum)
                self.prototypes[branch_idx, update_mask] = F.normalize(updated, dim=-1)
                updated_branches += 1

        aux = {
            "prototype_ready_ratio": self.initialized_ratio(),
            "prototype_updated_branches": self.prototypes.new_tensor(float(updated_branches)),
            "prototype_normal_tokens": self.prototypes.new_tensor(float(total_normal_tokens)),
        }
        return aux

    def branch_distances(self, branch_idx, img_token):
        valid_mask = self.initialized[branch_idx]
        if not valid_mask.any():
            return None

        patch_tokens = F.normalize(img_token[:, 1:, :], dim=-1)
        valid_prototypes = self.prototypes[branch_idx, valid_mask].to(dtype=patch_tokens.dtype)
        cosine = torch.matmul(patch_tokens, valid_prototypes.t()).clamp(-1.0 + self.eps, 1.0 - self.eps)
        nearest_cosine = cosine.max(dim=-1)[0]
        return torch.acos(nearest_cosine) / math.pi

    def loss(self, img_tokens, normal_mask, temperature=0.07, max_samples=0):
        if len(img_tokens) != self.num_branches:
            raise ValueError("Prototype bank branch count does not match img_tokens")

        flat_mask = normal_mask.reshape(normal_mask.shape[0], -1).bool()
        losses = []
        valid_branch_count = 0

        for branch_idx, img_token in enumerate(img_tokens):
            valid_mask = self.initialized[branch_idx]
            if not valid_mask.any():
                continue

            patch_tokens = F.normalize(img_token[:, 1:, :], dim=-1)
            branch_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1])
            branch_mask = flat_mask.reshape(-1)
            branch_tokens = branch_tokens[branch_mask]
            if branch_tokens.shape[0] < 1:
                continue

            branch_tokens = self._subsample(branch_tokens, max_samples=max_samples)
            valid_prototypes = self.prototypes[branch_idx, valid_mask].to(dtype=branch_tokens.dtype)
            logits = torch.matmul(branch_tokens, valid_prototypes.t()) / max(float(temperature), self.eps)
            targets = logits.detach().argmax(dim=-1)
            losses.append(F.cross_entropy(logits, targets))
            valid_branch_count += 1

        if len(losses) < 1:
            return None, {
                "prototype_loss": self.prototypes.new_tensor(0.0),
                "prototype_loss_branches": self.prototypes.new_tensor(0.0),
            }

        loss = torch.stack(losses).mean()
        aux = {
            "prototype_loss": loss.detach(),
            "prototype_loss_branches": self.prototypes.new_tensor(float(valid_branch_count)),
        }
        return loss, aux
