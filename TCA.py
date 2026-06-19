import numpy as np
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class FeatureLogitsWrapper(nn.Module):
    """
    Wraps an existing model so that forward(x) returns (feat, logits).
      feat  : [B, d] — input to the classifier head, captured via a pre-hook
      logits: [B, K]
    """
    def __init__(self, model: nn.Module, classifier_module_name: str):
        super().__init__()
        self.model = model
        self.classifier_module_name = classifier_module_name
        self._feat = None

        mods = dict(self.model.named_modules())
        if classifier_module_name not in mods:
            raise KeyError(f"'{classifier_module_name}' not found in model.named_modules()")
        # Use object.__setattr__ to avoid double-registering the module as a submodule,
        # which would cause duplicate keys in state_dict.
        object.__setattr__(self, 'classifier', mods[classifier_module_name])

        self._handle = self.classifier.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._feat = inputs[0]  # [B, d], kept on-device

    def forward(self, x):
        self._feat = None
        logits = self.model(x)
        feat = self._feat
        if feat is None:
            raise RuntimeError("Feature not captured. Check classifier_module_name.")
        return feat, logits

    def close(self):
        """Remove the forward hook (call when done to avoid memory leaks)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class TCA:
    """
    Test-time Correlation Alignment (TCA).

    Usage:
        tca = TCA(model, filter_K=20, W_num_iterations=20, W_lr=0.001)
        logits_corrected = tca.calculate(num_classes, embeddings, logits, proportion)
    """
    def __init__(self, model, filter_K=100, W_num_iterations=20, W_lr=0.001):
        self.classifier       = model.classifier
        self.filter_K         = filter_K
        self.W_num_iterations = W_num_iterations
        self.W_lr             = W_lr

    def calculate(self, num_classes, embeddings_arr, logits_arr, proportion_vector):
        """
        Args:
            num_classes      : int
            embeddings_arr   : [N, d] CPU tensor
            logits_arr       : [N, K] CPU tensor
            proportion_vector: [K]    CPU tensor  (relative class frequencies)
        Returns:
            [N, K] CPU tensor of TCA-corrected logits
        """
        self.supports   = embeddings_arr
        self.num_classes = num_classes
        self.labels     = logits_arr
        self.ent        = softmax_entropy(logits_arr)
        self.proportion = proportion_vector

        supports, _ = self._select_supports()

        # LinearCORAL on CPU; classifier inference chunked to GPU
        z_w = self._linear_coral(supports, embeddings_arr,
                                 num_iterations=self.W_num_iterations,
                                 learning_rate=self.W_lr)

        classifier_device = next(self.classifier.parameters()).device
        chunk_size = 768
        chunks = []
        for i in range(0, z_w.shape[0], chunk_size):
            chunk = z_w[i:i + chunk_size].to(classifier_device)
            chunks.append(self.classifier(chunk).detach().cpu())
        return torch.cat(chunks, dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _coral_loss(self, C_s, C_t):
        return torch.norm(C_s - C_t, p='fro') ** 2

    def _compute_covariance(self, X):
        X_mean = X.mean(dim=0, keepdim=True)
        return (X - X_mean).t() @ (X - X_mean) / (X.size(0) - 1)

    def _linear_coral(self, X_s, X_t, num_iterations=20, learning_rate=0.001):
        C_s    = self._compute_covariance(X_s)
        X_s_mean = X_s.mean(dim=0, keepdim=True)
        C_t    = self._compute_covariance(X_t)
        X_t_mean = X_t.mean(dim=0, keepdim=True)

        d = X_s.size(1)
        W = torch.eye(d, device=X_s.device, requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            C_t_transformed = W.T @ C_t.detach() @ W
            loss = self._coral_loss(C_t_transformed, C_s.detach())
            loss.backward()
            optimizer.step()

        return (X_t.detach() - X_t_mean) @ W.detach() + X_s_mean

    def _select_supports(self):
        ent_s    = self.ent
        y_hat    = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K

        if filter_K == -1:
            return self.supports, self.labels

        indices1 = torch.arange(len(ent_s), device=ent_s.device)
        indices  = []
        for i in range(self.num_classes):
            mask = (y_hat == i)
            _, order = torch.sort(ent_s[mask])
            k = math.floor(filter_K * self.proportion[i])
            indices.append(indices1[mask][order][:k])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels   = self.labels[indices]
        return self.supports, self.labels
