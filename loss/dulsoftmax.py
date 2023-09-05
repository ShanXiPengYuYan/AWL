import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy



class LossFunction(nn.Module):

    def __init__(self, embedding_size, nClasses, margin, scale, kl_lambda=1e-4, **kwargs):
        super(LossFunction, self).__init__()
        self.m = margin
        self.s = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, embedding_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        # self.ce = LabelSmoothingCrossEntropy()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        # lambda
        self.kl_lambda = kl_lambda
        print('Initialised AAMSoftmax margin %.3f scale %.3f, lambda %4.f' % (self.m, self.s, self.kl_lambda))

    def forward(self, mu, logvar, x, labels, **kwargs):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, labels)

        # dul regularization item
        kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
        kl_loss = kl_loss.sum(dim=1).mean()
        loss = loss + self.kl_lambda * kl_loss

        prec1 = accuracy(output.detach(), labels.detach(), topk=(1,))[0]

        return loss, prec1

