"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np

#################################################################################################
from MixTTA import get_mixtta_model_optim, get_mixtta_layer_names, spectral_projection
from MixTTA import set_record_feature, clear_feature_record, filter_feature_record
from sam import SAM
import TCA
#################################################################################################

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, args, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        
        #################################################################################################
        if self.args.plugin_mixtta:
            # get_mixtta_model_optim returns a plain SGD, but SAR needs SAM
            # (first_step/second_step); rebuild SAM over the plugin's parameters.
            self.model, _mixtta_optim = get_mixtta_model_optim(model, self.args)
            _mixtta_params = [p for g in _mixtta_optim.param_groups for p in g['params']]
            self.optimizer = SAM(_mixtta_params, torch.optim.SGD, lr=self.args.lr, momentum=0.9)
            self.mixtta_layer_names = get_mixtta_layer_names(self.model)
        else:
            self.model, self.optimizer = model, optimizer
        #################################################################################################

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            if self.args.Add_TCA:
                embeddings, outputs, ema, reset_flag = forward_and_adapt_sar(x, self.args, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            else:
                outputs, ema, reset_flag = forward_and_adapt_sar(x, self.args, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        if self.args.Add_TCA:
            return embeddings, outputs
        else:
            return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, args, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    ###############################################################
    if args.plugin_mixtta:
        set_record_feature(model, record_feature=True)
    ###############################################################

    # forward
    if args.Add_TCA:
        embeddings, outputs = model(x)
    else:
        outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    ###############################################################
    #if self.args.plugin_mixtta:
    #    filter_feature_record(self.model, filter_ids_1)
    #    spectral_projection(self.model, optimizer, self.args, self.mixtta_layer_names)
    #    clear_feature_record(self.model)
    ###############################################################

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    if args.Add_TCA:
        _, temp_output = model(x)
        entropys2 = softmax_entropy(temp_output)
    else:
        entropys2 = softmax_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()

    ###############################################################
    if args.plugin_mixtta and len(entropys2[filter_ids_2]) > 0:
        filter_feature_record(model, filter_ids_2)
        spectral_projection(model, optimizer, args, get_mixtta_layer_names(model))
        clear_feature_record(model)
    ###############################################################

    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    if args.Add_TCA:
        return embeddings, outputs, ema, reset_flag
    else:
        return outputs, ema, reset_flag


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
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
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
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"