import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy


class LossFunction(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """

    def __init__(self, embedding_size, nClasses, k=3, margin=0.25, scale=32, **kwargs):
        super(LossFunction, self).__init__()
        self.in_features = embedding_size
        self.out_features = nClasses
        self.s = scale
        self.m = margin
        self.K = k
        self.weight = nn.Parameter(torch.FloatTensor(nClasses * self.K, embedding_size),requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x, labels=None, ith_iter=None, **kwargs):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m


        phi = torch.where(cosine > 0, phi, cosine)


        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        loss = self.ce(output, labels)
        prec1 = accuracy(output.detach(), labels.detach(), topk=(1,))[0]

        return loss, prec1