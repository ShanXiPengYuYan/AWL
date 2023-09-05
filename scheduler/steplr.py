#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, step_size, lr_decay, last_epoch=-1, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay, last_epoch=last_epoch)
	print('Initialised step LR scheduler, step_size is {}, lr decay is {}'.format(step_size, lr_decay))
	return sche_fn
