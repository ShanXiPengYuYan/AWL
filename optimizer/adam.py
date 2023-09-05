#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised Adam optimizer, init_lr is {}, weight decay is {}'.format(lr, weight_decay))

	return torch.optim.Adam(parameters, lr = lr, weight_decay = weight_decay);
