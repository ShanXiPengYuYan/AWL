#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, momentum, **kwargs):

	print('Initialised SGD optimizer init lr is {}, weight decay is {}, momentum is {}'.format(lr, weight_decay, momentum))

	return torch.optim.SGD(parameters, lr = lr, momentum = momentum, weight_decay=weight_decay);
