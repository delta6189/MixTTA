from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Layer Modules
# ──────────────────────────────────────────────────────────────────────────────

class FCLinear(nn.Module):
    """Full-rank affine layer (FC variant of the MixTTA plugin)."""
    def __init__(self, channels, args, norm=None):
        super().__init__()
        self.channels = channels
        self.args = args
        self.norm = norm

        affine_scale = norm.weight.detach().clone() if (hasattr(norm, 'weight') and norm.weight is not None) else torch.ones(channels)
        affine_bias  = norm.bias.detach().clone()   if (hasattr(norm, 'bias')  and norm.bias  is not None) else torch.zeros(channels)

        if self.norm is not None:
            self.norm.affine = False
            self.norm.weight = None
            self.norm.bias   = None

        self.matrix      = nn.Parameter(torch.diag(affine_scale))
        self.affine_bias = nn.Parameter(affine_bias)

    def forward(self, x):
        matrix = self.matrix
        if self.args.model in ['resnet50_bn_torch', 'resnet50_gn_timm']:
            B, C, H, W = x.shape
            x_out = x.flatten(2).transpose(1, 2)       # (B, HW, C)
            x_out = F.linear(x_out, matrix.T)          # (B, HW, C)
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
            x_out = x_out + self.affine_bias.view(1, -1, 1, 1)
        else:  # vitbase_timm: (B, N, C)
            x_out = F.linear(x, matrix.T) + self.affine_bias
        return x_out


class LoRAFCLinear(nn.Module):
    """
    Low-rank (LoRA-style) affine layer — the main MixTTA plugin layer.

    Forward:  out = affine_branch(norm(x)) + AB_branch(norm(x))
      affine_branch: standard scale+bias (initialized from the original norm)
      AB_branch:     (coeff * alpha/r) * norm_stop(x A) B   where A:(C,r), B:(r,C)
    """
    def __init__(self, channels, alpha, r, args, norm=None):
        super().__init__()
        self.channels = channels
        self.r = r
        self.args = args
        self.coeff = alpha / r

        if isinstance(norm, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            self.affine_scale = nn.Parameter(norm.weight.detach().clone())
            self.affine_bias  = nn.Parameter(norm.bias.detach().clone())
            self.norm = norm
            self.norm.affine = False
            self.norm.weight = None
            self.norm.bias   = None
            self.norm.elementwise_affine = False
        else:
            self.register_buffer('affine_scale', torch.ones(channels))
            self.register_buffer('affine_bias',  torch.zeros(channels))
            self.norm = nn.Identity()

        # Initialise A with the requested scheme; B is always zero (standard LoRA)
        matrix_A = torch.empty(channels, r)
        if args.init_type == 'xavier':
            nn.init.xavier_normal_(matrix_A)
        elif args.init_type == 'kaiming':
            nn.init.kaiming_uniform_(matrix_A, a=math.sqrt(5))
        elif args.init_type == 'orthogonal':
            nn.init.orthogonal_(matrix_A)
        else:
            raise ValueError(f"Unknown init_type: '{args.init_type}'. "
                             f"Choose from: xavier, kaiming, orthogonal.")

        self.matrix_A = nn.Parameter(matrix_A)
        self.matrix_B = nn.Parameter(torch.zeros(r, channels))

        # Feature recording (for Spectral Projection)
        self.feature        = None
        self.record_feature = False

        # Logging accumulators (cleared after each print interval)
        self.mu_norm_list = []
        self.rho_list     = []
        self.kappa_list   = []
        self.lambda_list  = []
        self.N            = 0

    def forward(self, x):
        x_norm = self.norm(x)
        return self.AB_branch(x_norm) + self.affine_branch(x_norm)

    def AB_branch(self, x):
        matrix_A = self.matrix_A
        matrix_B = self.matrix_B

        if self.args.decouple_proj:
            matrix_B = self.decoupling_project_(matrix_A, matrix_B)

        if self.args.model in ['resnet50_bn_torch', 'resnet50_gn_timm']:
            """
            B, C, H, W = x.shape
            x_out = x.flatten(2).transpose(1, 2)       # (B, HW, C)
            x_out = F.linear(x_out, matrix_A.T)        # (B, HW, r)

            if self.record_feature:
                self.feature = x_out
                self._logging(x_out)

            x_out = F.linear(x_out, matrix_B.T)        # (B, HW, C)
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
            """

            B, C, H, W = x.shape
            matrix_A = matrix_A.transpose(0, 1).contiguous().view(self.r, self.channels, 1, 1) # (C, r) -> (r, C, 1, 1)
            matrix_B = matrix_B.transpose(0, 1).contiguous().view(self.channels, self.r, 1, 1) # (r, C) -> (C, r, 1, 1)

            x_out = F.conv2d(x, matrix_A)              # (B, C, H, W) -> (B, r, H, W)  [raw z]
            if self.record_feature:
                self.feature = x_out.view(B, self.r, -1).transpose(1, 2)  # (B, r, HW) -> (B, HW, r)
                self._logging(self.feature)
            x_out = F.conv2d(x_out, matrix_B)          # (B, r, H, W) -> (B, C, H, W)

        else:  # vitbase_timm: (B, N, C)
            B, N, C = x.shape
            x_out = F.linear(x, matrix_A.T)            # (B, N, r)

            if self.record_feature:
                self.feature = x_out
                self._logging(x_out)

            x_out = F.linear(x_out, matrix_B.T)        # (B, N, C)

        return x_out * self.coeff

    def affine_branch(self, x_norm):
        if self.args.model in ['resnet50_bn_torch', 'resnet50_gn_timm']:
            return x_norm * self.affine_scale.view(1, -1, 1, 1) + self.affine_bias.view(1, -1, 1, 1)
        else:
            return x_norm * self.affine_scale + self.affine_bias

    def _logging(self, feature):
        """Accumulate per-batch statistics for the diagnostic print."""
        feature = feature.detach()
        B, T, r = feature.shape
        mu  = feature.mean(dim=1, keepdim=True)             # (B, 1, r)
        zc  = feature - mu
        trC = zc.pow(2).sum(dim=(1, 2)) / (T - 1)          # (B,)
        rho = mu.squeeze(1).pow(2).sum(dim=1) / (trC + 1e-6)

        Cov  = (zc.transpose(1, 2) @ zc) / (T - 1)
        epsI = 1e-6 * torch.eye(r, device=Cov.device, dtype=Cov.dtype).unsqueeze(0)
        Cov  = Cov + epsI
        try:
            evals = torch.linalg.eigvalsh(Cov)
            kappa = evals[:, -1] / evals[:, 0]
            self.mu_norm_list.append(mu.squeeze(1).norm(dim=1).sum(dim=0))
            self.rho_list.append(rho.sum(dim=0))
            self.kappa_list.append(kappa.sum(dim=0))
            self.lambda_list.append(evals.sum(0))   # (r,) sum over batch
        except Exception:
            pass
        self.N += B

    def decoupling_project_(self, A, B, eps=1e-12):
        """
        Decoupling Projection: remove the component of each B-row that is
        parallel to the corresponding A-column, so that diag(A @ B^T) = 0.
        A: (C, r), B: (r, C) — returns projected B (r, C), does not modify in-place.
        """
        a = A                                  # (C, r)
        b = B.transpose(0, 1).contiguous()     # (C, r)
        denom = (a * a).sum(dim=1, keepdim=True) + eps
        coeff = (a * b).sum(dim=1, keepdim=True) / denom
        b = b - (coeff * a).detach()
        return b.transpose(0, 1)               # (r, C)


class MixTTANorm(nn.Module):
    """Wrapper that replaces a norm layer with a FCLinear or LoRAFCLinear."""
    def __init__(self, channels, norm_layer, args):
        super().__init__()
        self.channels   = channels
        self.norm_layer = norm_layer

        if args.layer_type == 'FC':
            self.linear = FCLinear(channels, args, norm=self.norm_layer)
        elif args.layer_type == 'LoRAFC':
            self.linear = LoRAFCLinear(channels, args.alpha, args.r, args, norm=self.norm_layer)
        else:
            raise ValueError(f"Unknown layer_type: '{args.layer_type}'. Choose from: FC, LoRAFC.")

    def forward(self, x):
        return self.linear(x)


# ──────────────────────────────────────────────────────────────────────────────
#  Plugin: model modification & optimiser construction
# ──────────────────────────────────────────────────────────────────────────────

def _get_layer_name_list(model_name, target_blocks, target_layers, target_norms):
    name_list = []
    if model_name in ['resnet50_bn_torch', 'resnet50_gn_timm']:
        for block_id in target_blocks:
            if int(block_id) == 0:
                name_list.append('bn1')
                continue
            for layer_id in target_layers:
                for norm_id in target_norms:
                    if int(norm_id) < 4:
                        norm_name = f'bn{norm_id}'
                    else:
                        norm_name = 'downsample.1'
                    layer_name = f'layer{block_id}.{layer_id}.{norm_name}'
                    name_list.append(layer_name)
    elif model_name in ['vitbase_timm']:
        for block_id in target_blocks:
            for norm_id in target_norms:
                name_list.append(f'blocks.{block_id}.norm{norm_id}')
    return name_list


def _wrap_norm_layers(model, target_layers_name, args):
    for full_name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            continue
        if full_name not in target_layers_name:
            continue
        if isinstance(module, MixTTANorm):
            continue
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
        elif isinstance(module, nn.GroupNorm):
            num_channels = module.num_channels
        else:  # LayerNorm
            num_channels = module.normalized_shape[0]

        parent, attr = _get_parent_module(model, full_name)
        setattr(parent, attr, MixTTANorm(num_channels, module, args))


def _get_parent_module(model, full_name):
    parts = full_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def collect_params(model, tent_layer_list, mixtta_layer_list):
    """
    Returns (base_params, base_names, mixtta_params, mixtta_names).
      base_params   — affine scale/bias inside MixTTANorm (optimised with base LR)
      mixtta_params — matrix_A / matrix_B inside LoRAFCLinear
    """
    base_params, base_names     = [], []
    mixtta_params, mixtta_names = [], []

    for nm, m in model.named_modules():
        if isinstance(m, MixTTANorm):
            if nm in mixtta_layer_list:
                for np_, p in m.named_parameters():
                    full = f'{nm}.{np_}'
                    if 'matrix_A' in np_ or 'matrix_B' in np_ or 'matrix' in np_:
                        mixtta_params.append(p)
                        mixtta_names.append(full)
                    elif 'affine_bias' in np_ or 'affine_scale' in np_:
                        base_params.append(p)
                        base_names.append(full)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if nm in tent_layer_list:
                for np_, p in m.named_parameters():
                    full = f'{nm}.{np_}'
                    if 'weight' in np_ or 'bias' in np_:  # weight is scale, bias is shift
                        base_params.append(p)
                        base_names.append(full)


    print('Tent (affine) params:')
    print(base_names)
    print('MixTTA (LoRA) params:')
    print(mixtta_names)
    return base_params, base_names, mixtta_params, mixtta_names


def get_mixtta_model_optim(model, args):
    """Inject MixTTA plugin layers and return (modified model, new optimizer)."""
    tent_layer_names   = _get_layer_name_list(args.model, args.tent_target_blocks,
                                              args.tent_target_layers, args.tent_target_norms)
    mixtta_layer_names = _get_layer_name_list(args.model, args.mixtta_target_blocks,
                                              args.mixtta_target_layers, args.mixtta_target_norms)

    _wrap_norm_layers(model, mixtta_layer_names, args)
    base_params, base_names, mixtta_params, mixtta_names = collect_params(
        model, tent_layer_names, mixtta_layer_names)

    print(f"Tent param count : {len(base_params)} tensors / "
          f"{sum(p.numel() for p in base_params)} elements")
    print(f"MixTTA param count: {len(mixtta_params)} tensors / "
          f"{sum(p.numel() for p in mixtta_params)} elements")

    model.train()
    model.cuda()

    if args.exp_type == 'mix_shifts':
        optimizer = torch.optim.SGD([
            {'params': base_params,   'lr': args.lr * args.lr_coeff},
            {'params': mixtta_params, 'lr': args.lr},
        ], momentum=0.9)
    else:
        optimizer = torch.optim.SGD(
            base_params + mixtta_params,
            lr=args.lr * args.lr_coeff,
            momentum=0.9)

    return model, optimizer


# ──────────────────────────────────────────────────────────────────────────────
#  Feature recording helpers (used by recap_plpd for Spectral Projection)
# ──────────────────────────────────────────────────────────────────────────────

def set_record_feature(model, record_feature: bool):
    for _, m in model.named_modules():
        if isinstance(m, MixTTANorm) and isinstance(m.linear, LoRAFCLinear):
            m.linear.record_feature = record_feature


def clear_feature_record(model):
    for _, m in model.named_modules():
        if isinstance(m, MixTTANorm) and isinstance(m.linear, LoRAFCLinear):
            m.linear.record_feature = False
            m.linear.feature        = None


def filter_feature_record(model, filter_ids):
    for _, m in model.named_modules():
        if isinstance(m, MixTTANorm) and isinstance(m.linear, LoRAFCLinear):
            m.linear.feature = m.linear.feature[filter_ids]


def get_mixtta_layer_names(model) -> List[str]:
    return [name for name, m in model.named_modules()
            if isinstance(m, MixTTANorm) and isinstance(m.linear, LoRAFCLinear)]


def clear_logging_stats(model):
    """Reset per-interval logging accumulators to prevent GPU memory growth."""
    for _, m in model.named_modules():
        if isinstance(m, MixTTANorm) and isinstance(m.linear, LoRAFCLinear):
            m.linear.mu_norm_list = []
            m.linear.rho_list     = []
            m.linear.kappa_list   = []
            m.linear.lambda_list  = []
            m.linear.N            = 0


# ──────────────────────────────────────────────────────────────────────────────
#  Spectral Projection  (gradient projection onto the spectral subspace)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _top1_u_per_sample(Z: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Power-iteration estimate of the leading eigenvector of Cov[Z] per sample.
    Z: [B, T, r]  →  U: [B, r]
    """
    Zc = Z - Z.mean(dim=1, keepdim=True)            # centre tokens
    U  = torch.randn(Z.size(0), Z.size(2), device=Z.device, dtype=Z.dtype)
    U  = U / (U.norm(dim=1, keepdim=True) + eps)
    for _ in range(iters):
        Zv = (Zc * U[:, None, :]).sum(dim=-1)       # [B, T]
        Y  = torch.einsum('btr,bt->br', Zc, Zv)    # [B, r]
        U  = Y / (Y.norm(dim=1, keepdim=True) + eps)
    return U                                        # [B, r]


@torch.no_grad()
def spectral_project_grad(A: torch.Tensor, B: torch.Tensor,
                          Z: torch.Tensor, optimizer, args):
    """
    Spectral Projection: orthogonally project A.grad and B.grad away from
    the leading spectral direction of the subspace feature Z.

    A: [C, r], B: [r, C], Z: [B, T, r]
    Projection strength: args.eta ∈ [0, 1]  (1 = full removal)
    """
    if A.grad is None or B.grad is None or Z.size(0) < 1:
        return

    r = Z.size(2)
    eps = 1e-6

    u1     = _top1_u_per_sample(Z)                 # [B, r]
    u1_dir = u1 / (u1.norm(dim=1, keepdim=True) + eps)

    # Rank-1 projection matrix averaged over the batch
    U = u1_dir.unsqueeze(2)                        # [B, r, 1]
    P = (U @ U.transpose(1, 2)).mean(0)            # [r, r]

    # Project gradients: g ← g − η (g P) for A,  g ← g − η (P g) for B
    A.grad = A.grad - args.eta * (A.grad @ P)
    B.grad = B.grad - args.eta * (P @ B.grad)

    # Apply the same projection to the SGD momentum buffer
    inner_opt = getattr(optimizer, 'base_optimizer', optimizer)
    for group in inner_opt.param_groups:
        for p in group['params']:
            st = inner_opt.state.get(p)
            if st is None or 'momentum_buffer' not in st:
                continue
            v = st['momentum_buffer']
            if v is None:
                continue
            if p is A and v.ndim == 2 and v.size(1) == r:
                st['momentum_buffer'] = v - args.eta * (v @ P)
            elif p is B and v.ndim == 2 and v.size(0) == r:
                st['momentum_buffer'] = v - args.eta * (P @ v)


def spectral_projection(model, optimizer, args, target_layer_names):
    """Apply Spectral Projection to all MixTTA layers in target_layer_names."""
    for name, m in model.named_modules():
        if (isinstance(m, MixTTANorm)
                and isinstance(m.linear, LoRAFCLinear)
                and name in target_layer_names):
            spectral_project_grad(
                m.linear.matrix_A, m.linear.matrix_B,
                m.linear.feature,  optimizer, args)


# ──────────────────────────────────────────────────────────────────────────────
#  Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _decoupling_project(A, B, eps=1e-12):
    """Stateless version of decoupling_project_ (used in print_model_stats)."""
    a = A
    b = B.transpose(0, 1).contiguous()
    denom = (a * a).sum(dim=1, keepdim=True) + eps
    coeff = (a * b).sum(dim=1, keepdim=True) / denom
    return (b - (coeff * a).detach()).transpose(0, 1)


def _format_val(val):
    val = float(val)
    return f'{val:.1e}' if abs(val) < 0.01 else f'{val:.2f}'


def print_model_stats(classifier, args):
    ft_headers, ft_data, widths = [], [], []

    for name, m in classifier.named_modules():
        if not isinstance(m, MixTTANorm):
            continue

        if isinstance(m.linear, FCLinear):
            matrix       = m.linear.matrix
            affine_scale = torch.diagonal(matrix)
            affine_bias  = m.linear.affine_bias
            channels     = affine_scale.size(0)
            rank         = torch.linalg.matrix_rank(matrix).item()
            _, S, _      = torch.linalg.svd(matrix, full_matrices=False)

            sigma_range       = f'{_format_val(S.max())} ~ {_format_val(S.min())}'
            affine_scale_rng  = f'{_format_val(affine_scale.max())} ~ {_format_val(affine_scale.min())}'
            affine_bias_rng   = f'{_format_val(affine_bias.max())} ~ {_format_val(affine_bias.min())}'
            grad_m      = round(matrix.grad.data.norm(2).item(), 2) if matrix.grad is not None else 'None'
            grad_scale  = round(affine_scale.grad.data.norm(2).item(), 2) if affine_scale.grad is not None else 'None'
            grad_bias   = round(affine_bias.grad.data.norm(2).item(), 2) if affine_bias.grad is not None else 'None'
            norm_m      = round(torch.norm(matrix, p='fro').item(), 3)

            ft_headers = ['Layer', 'C', 'Rank', 'Affine Scale', 'Affine Bias', 'Sigma', '∇Scale', '∇Bias', '∇M', '||M||']
            widths     = [35, 5, 5, 15, 20, 15, 10, 10, 10, 10]
            ft_data.append([name, channels, rank, affine_scale_rng, affine_bias_rng,
                            sigma_range, grad_scale, grad_bias, grad_m, norm_m])

        elif isinstance(m.linear, LoRAFCLinear):
            affine_scale = m.linear.affine_scale
            affine_bias  = m.linear.affine_bias
            matrix_A     = m.linear.matrix_A
            matrix_B     = m.linear.matrix_B
            I_r          = torch.eye(matrix_A.size(1), device=matrix_A.device)
            channels     = affine_scale.size(0)

            if args.decouple_proj:
                matrix_B = _decoupling_project(matrix_A, matrix_B)

            weight    = matrix_A @ matrix_B
            diag      = torch.diag(torch.diagonal(weight))
            off_diag  = weight - diag
            diag_abs     = round(torch.trace(torch.abs(diag)).item(), 3)
            off_diag_abs = round(torch.norm(off_diag, p=1).item(), 3)

            try:
                rank = torch.linalg.matrix_rank(weight).item()
                S    = torch.linalg.svdvals(weight)
            except Exception:
                rank = 0
                S    = torch.zeros(1)

            sigma_rng      = f'{_format_val(S.max())} ~ {_format_val(S.min())}'
            affine_rng     = f'{_format_val(affine_scale.max())} ~ {_format_val(affine_scale.min())}'
            _raw_A = m.linear.matrix_A
            _raw_B = m.linear.matrix_B
            grad_A = round(_raw_A.grad.data.norm(2).item(), 2) if _raw_A.grad is not None else 'None'
            grad_B = round(_raw_B.grad.data.norm(2).item(), 2) if _raw_B.grad is not None else 'None'

            N            = m.linear.N
            mu_norm_list = m.linear.mu_norm_list
            kappa_list   = m.linear.kappa_list
            lambda_list  = m.linear.lambda_list
            mu_norm_mean = round(sum(mu_norm_list).item() / N, 3) if N > 0 and mu_norm_list else 0.0
            kappa_mean   = round(sum(kappa_list).item()   / N, 3) if N > 0 and kappa_list   else 0.0
            if N > 0 and lambda_list:
                lambdas_mean = sum(lambda_list) / N          # (r,) mean eigenvalues
                lambdas = list(reversed([round(lam, 3) for lam in lambdas_mean.tolist()]))
            else:
                lambdas = []

            ft_headers = ['Layer', 'C', 'R', 'affine_scale', 'sigma',
                          '|D|', '|OffD|', 'Grad_A', 'Grad_B', 'norm_mu', 'kappa', 'lambdas']
            widths     = [25, 5, 5, 15, 18, 8, 8, 8, 8, 8, 8, 30]
            ft_data.append([name, channels, rank, affine_rng, sigma_rng,
                            diag_abs, off_diag_abs, grad_A, grad_B,
                            mu_norm_mean, kappa_mean, lambdas])

    if not ft_data or not ft_headers:
        return

    formatted_header = ' | '.join(f'{n:<{w}}' for n, w in zip(ft_headers, widths))
    print(formatted_header)
    print('-' * (sum(widths) + 3 * (len(widths) - 1)))
    for row in ft_data:
        print(' | '.join(f'{str(v):<{w}}' for v, w in zip(row, widths)))
    print()
