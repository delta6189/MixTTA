"""
Copyright to ReCAP Authors, ICML 2025 Poster.
built upon on SAR and DeYO code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from einops import rearrange
#################################################################################################
from MixTTA import get_mixtta_model_optim, get_mixtta_layer_names, spectral_projection
from MixTTA import set_record_feature, clear_feature_record, filter_feature_record
import TCA
#################################################################################################

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

class ReCAP(nn.Module):
    def __init__(self, model, args, optimizer, sigmas, batch_size, steps=1, episodic=False, margin=0.8*math.log(1000), \
        reset_constant_em=0.2, margin_L0 = 0.8 * math.log(1000), weight_reg = 0.5, reweight_threshold = 3.0, weight_tau = 1.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ReCAP requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.batch_size = batch_size

        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme, follow SAR
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        #################################################################################################
        self.args = args

        # Step 1: MixTTA plugin — must run on the raw model so layer names match
        if args.plugin_mixtta:
            self.model, self.optimizer = get_mixtta_model_optim(model, args)
        else:
            self.model, self.optimizer = model, optimizer

        # Step 2: LinearTCA — wrap on top of the (possibly MixTTA-modified) model
        if args.Add_TCA:
            if not isinstance(self.model, TCA.FeatureLogitsWrapper):
                classifier_name = 'fc' if hasattr(self.model, 'fc') else 'head'
                self.model = TCA.FeatureLogitsWrapper(self.model, classifier_name)
            self._tca_embeddings = []
            self._tca_logits = []

        # Step 3: collect layer names AFTER all wrapping so names match at projection time
        if args.plugin_mixtta:
            self.mixtta_layer_names = get_mixtta_layer_names(self.model)
        #################################################################################################

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.margin = margin # margin \tau_RE in Eqn. (9)
        self.margin_L0 = margin_L0 # L_0 in Eqn. (9)
        self.weight_reg = weight_reg
        self.reweight_threshold = reweight_threshold
        self.weight_tau = weight_tau
        self.sigma_t = sigmas

        if args.Add_TCA:
            self.W = self.model.classifier.weight   # FeatureLogitsWrapper.classifier
        else:
            try:
                self.W = self.model.fc.weight       # ResNet
            except AttributeError:
                self.W = self.model.head.weight     # ViT
        self.W_cpu = self.W.cpu()
        self._refresh_prob_aug()


    def _refresh_prob_aug(self, scale = 0.1):
        with torch.no_grad():
            sigma_t = self.sigma_t.view(1, 1, -1)
            region = sigma_t * self.weight_tau / scale
            sqrt_region = torch.sqrt(region).cpu()
            diff = (self.W_cpu.unsqueeze(0) - self.W_cpu.unsqueeze(1)) * sqrt_region
            self.prob_aug = torch.exp(0.5 * torch.einsum('ijb,ijb->ij', diff, diff))
            self.prob_aug = self.prob_aug.cuda()
            self.normW = 0.1 / 2 * (scale ** 2)  * (torch.norm(self.W, dim=1) ** 2)

    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def L_RE(self, x: torch.Tensor) -> torch.Tensor:
        """Implicit augmentation using gaussian noise. speed up"""
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        prob = (x + self.normW).softmax(1)
        return (-prob * torch.log(prob_anchor) + prob * torch.log(prob_aug)).sum(1)

    def L_RI(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        return (prob_anchor * torch.log(prob_aug)).sum(1)


    def _fwd(self, x, record_feat=True):
        """Internal forward: always returns (embeddings_or_None, logits)."""
        set_record_feature(self.model, record_feature=record_feat)
        if self.args.Add_TCA:
            emb, out = self.model(x)
            return emb, out
        else:
            out = self.model(x)
            return None, out

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_recap(self, x, ema):

        self.optimizer.zero_grad()
        emb, outputs = self._fwd(x, record_feat=True)
        
        L_RE = self.L_RE(outputs)
        L_RI = self.L_RI(outputs)

        filter_ids_1 = torch.where(L_RE < self.margin)

        if filter_ids_1[0].numel() == 0:
            return emb, outputs, ema, False

        L_RE = L_RE[filter_ids_1]
        L_RI = L_RI[filter_ids_1]

########################################  plpd  ####################################################
        x_prime = x[filter_ids_1]
        x_prime = x_prime.detach()
        
        if self.args.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.args.occlusion_size, self.args.occlusion_size)
            x_prime[:, :, self.args.row_start:self.args.row_start+self.args.occlusion_size,
                    self.args.column_start:self.args.column_start+self.args.occlusion_size] = occlusion_window
        elif self.args.aug_type == 'patch':
            patch_len = self.args.patch_len
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//patch_len)*patch_len,(x.shape[-1]//patch_len)*patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
            x_prime = resize_o(x_prime)
        elif self.args.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        
        with torch.no_grad():
            _, outputs_prime = self._fwd(x_prime, record_feat=False)
        
        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
        
        plpd_threshold = 0.2
        filter_ids_2 = torch.where(plpd > plpd_threshold)

        if filter_ids_2[0].numel() == 0:
            return emb, outputs, ema, False

########################################  add reweighting coefficient  ####################################################

        L_RE = L_RE[filter_ids_2]
        L_RI = L_RI[filter_ids_2]

        RE = L_RE.detach().clone()
        RI = L_RI.detach().clone()

        coeff = torch.min(torch.exp(self.margin_L0 - RE), torch.tensor(self.reweight_threshold))
        loss = (L_RE + self.weight_reg * L_RI).mul(coeff).mean(0)

########################################  add reweighting coefficient  ####################################################

        if not np.isnan(loss.item()):
            ema = update_ema(ema, loss.item() / 2) # record moving average loss values for model recovery
        
        loss.backward()
        ###############################################################
        if self.args.plugin_mixtta and self.args.eta > 0:
            filter_feature_record(self.model, filter_ids_1)
            filter_feature_record(self.model, filter_ids_2)
            spectral_projection(self.model, self.optimizer, self.args, self.mixtta_layer_names)
            clear_feature_record(self.model)
        ###############################################################
        self.optimizer.step()

        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return emb, outputs, ema, reset_flag

    def forward(self, x, no_adapt=False):
        if self.episodic:
            self.reset()

        if no_adapt:
            out = self.model(x)
            return out[1] if self.args.Add_TCA else out

        for _ in range(self.steps):
            emb, outputs, ema, reset_flag = self.forward_and_adapt_recap(x, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        # Accumulate TCA buffer on CPU to avoid GPU memory explosion
        if self.args.Add_TCA and emb is not None:
            self._tca_embeddings.append(emb.detach().cpu())
            self._tca_logits.append(outputs.detach().cpu())

        return outputs  # always single output regardless of Add_TCA

    def compute_tca_output(self, labels_arr):
        """Apply post-hoc LinearTCA to all accumulated embeddings.

        Args:
            labels_arr: list of target tensors collected per batch in main loop.
        Returns:
            Tensor of shape [N, C] with TCA-corrected logits (CPU, detached).
        """
        # Cat on CPU (buffers already stored on CPU), then move to GPU only for classifier call
        embeddings_all = torch.cat(self._tca_embeddings, dim=0)   # CPU
        logits_all = torch.cat(self._tca_logits, dim=0)           # CPU
        self.reset_tca_buffer()  # free list memory immediately after cat

        labels_all = torch.cat(labels_arr).cpu()
        label_counts = torch.bincount(labels_all)
        num_classes = len(label_counts)

        proportion_vector = torch.zeros(num_classes)
        proportion_vector[0] = 1.0
        base_count = max(label_counts[0].item(), 1)
        for i in range(1, num_classes):
            proportion_vector[i] = label_counts[i].item() / base_count

        tca_ = TCA.TCA(self.model, filter_K=self.args.filter_K_TCA,
                       W_num_iterations=self.args.W_num_iterations, W_lr=self.args.W_lr)
        return tca_.calculate(num_classes, embeddings_all, logits_all, proportion_vector).detach().cpu()

    def reset_tca_buffer(self):
        """Clear accumulated embeddings/logits (call between corruptions)."""
        self._tca_embeddings = []
        self._tca_logits = []

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None




def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def collect_params_dict(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return a dictionary of parameters with their names as keys.
    Note: other choices of parameterization are possible!
    """
    params_dict = {}
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params_dict[f"{nm}.{np}"] = p

    return params_dict




def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with ReCAP."""
    # train mode, because ReCAP optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what ReCAP updates
    model.requires_grad_(False)
    # configure norm for ReCAP updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with ReCAP."""
    is_training = model.training
    assert is_training, "ReCAP needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "ReCAP needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "ReCAP should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "ReCAP needs normalization layer parameters for its optimization"