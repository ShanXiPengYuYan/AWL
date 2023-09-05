#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
	def __init__(self, embedding_size, nClasses, total_iter=None, **kwargs):
		super(LossFunction, self).__init__()
		self.test_normalize = True
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, embedding_size), requires_grad=True)
		self.ce = nn.CrossEntropyLoss()
		self.fc = nn.Linear(embedding_size, nClasses)

		self.t = 0.2
		# make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
		self.eps = 1e-3
		self.t_alpha = 0.98
		self.register_buffer('batch_mean', torch.zeros(1) * (20))
		if total_iter != None:
			self.total_iter = total_iter
		print('Initialised awl_Softmax Loss')

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
		output =  F.linear(x, self.weight)
		cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, labels.view(-1, 1), 1)
		output = (one_hot * output) + ((1.0 - one_hot) * cosine)

		# print(m_sita_grad.shape)
		log_p = F.log_softmax(output, dim=-1)
		pt = torch.exp(log_p)

		labels = labels.view(-1, 1)
		pt = pt.gather(1, labels)

		m_sita_grad = margin_scaler * (1 - cosine) + 2
		lambda_run = ith_iter / self.total_iter
		weights = lambda_run * m_sita_grad + (1 - lambda_run) * 1
		ce_loss = -torch.log(pt)
		loss = weights * ce_loss
		loss = loss.mean()
		prec1 = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
		return loss, prec1