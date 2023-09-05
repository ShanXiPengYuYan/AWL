import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
import math

class LossFunction(nn.Module):
    def __init__(self, embedding_size, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.m = 1.35
        self.s = 64
        self.in_features = embedding_size
        self.out_features = nClasses
        self.fc = nn.Linear(embedding_size, nClasses, bias=False)
        self.eps = 1e-12

    print('Initialised ASoftmax')

    def forward(self, x, labels=None, **kwargs):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        out = torch.cos(torch.clip(self.m*torch.acos(cosine), self.eps, math.pi-self.eps))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        out = (one_hot * out) + ((1.0 - one_hot) * cosine)
        out = out * self.s

        loss = self.ce(out, labels)

        prec1   = accuracy(out.detach(), labels.detach(), topk=(1,))[0]
        return loss, prec1

