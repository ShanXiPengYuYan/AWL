import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math

from torch.autograd import Variable

from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, embedding_size, nClasses, margin=0.3, scale=15, easy_margin=True, total_iter=None, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = embedding_size
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, embedding_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.eps = 1e-3
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.t = 0.2
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.t_alpha = 0.98
        self.register_buffer('batch_mean', torch.zeros(1) * (20))

        if total_iter != None:
            self.total_iter = total_iter

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))


    def forward(self, x, labels=None, ith_iter=None, **kwargs):

        norms = torch.norm(x, dim=1, keepdim=True)  # l2 norm
        safe_norms = torch.clip(norms, min=0.001, max=200)
        safe_norms = safe_norms.clone().detach()
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean

        ones = torch.ones_like(safe_norms)
        margin_scaler = -1 * ones
        margin_scaler[safe_norms > self.batch_mean] = 1

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # if self.easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        log_p = F.log_softmax(output, dim=-1)
        pt = torch.exp(log_p)

        labels = labels.view(-1,1)
        pt = pt.gather(1, labels)
        m_sita_grad = margin_scaler * (1 - cosine) + 2

        lambda_run = ith_iter / self.total_iter
        weights = lambda_run * m_sita_grad + (1 - lambda_run) * 1
        ce_loss = -torch.log(pt)
        loss = weights * ce_loss
        loss = loss.mean()
        prec1   = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
        return loss, prec1

