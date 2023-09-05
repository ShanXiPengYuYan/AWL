import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, embedding_size, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = embedding_size
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, embedding_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, labels=None, **kwargs):

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, labels)
        prec1   = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
        return loss, prec1