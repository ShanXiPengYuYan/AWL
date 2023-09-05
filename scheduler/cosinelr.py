#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, total_iter, min_lr=1e-8, last_epoch=-1, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, verbose=False, eta_min=min_lr)
	print('Initialised CosineAnnealing LR scheduler, T_max is {}'.format(total_iter))
	return sche_fn
