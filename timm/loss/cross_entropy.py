""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class rein_LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(rein_LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor,reinforce:float=0,indices:np.array=np.array([])) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        coeffs = torch.ones_like(target).float()
        coeffs[indices] += reinforce
        coeffs /= (indices.sum() * reinforce + len(target))
        loss = loss * coeffs
        return loss.sum()


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
class rein_CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,**kwargs):
        super(rein_CrossEntropyLoss,self).__init__()
    def forward(self,input: torch.Tensor, target: torch.Tensor,reinforce:float=0.0,indices:np.array=np.array([])) -> torch.Tensor:
        loss = F.cross_entropy(input, target, weight=self.weight,ignore_index=self.ignore_index, reduction="none")
        coeffs = torch.ones_like(target).float()
        coeffs[indices] += reinforce
        coeffs /= (indices.sum() * reinforce + len(target))
        loss = loss * coeffs
        return loss.sum()

class rein_SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(rein_SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor,reinforce:float=0.0,indices:np.array=np.array([])) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        # coeffs = torch.ones_like(target).float()
        coeffs = torch.ones([len(target)]).to(target.device)
        coeffs[indices] += reinforce
        coeffs /= (indices.sum() * reinforce + len(target))
        loss = loss * coeffs
        return loss.sum()

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
