#!/usr/bin/python
#-*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, itertools, shutil, importlib
from tqdm import tqdm
import soundfile as sf
import DataLoader
from utils import FbankAug, warm_up_lr
from NISQAmaster.nisqa.NISQA_model import nisqaModel
class SpeakerNet(nn.Module):

    def __init__(self, args, config):
        super(SpeakerNet, self).__init__()
        # self.nisqa = nisqaModel(vars(args))

        ## model
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        #SpeakerNetModel = importlib.import_module('models.'+config.MODEL.NAME).__getattribute__('MainModel')
        SpeakerNetModel = importlib.import_module('models.'+config.MODEL.NAME).__getattribute__('MainModel')

        specaug = None
        if config.AUG.FLAG:
            specaug = FbankAug(freq_mask_width=config.AUG.SPECAUG.FREQ_MASK_WIDTH, time_mask_width=config.AUG.SPECAUG.TIME_MASK_WIDTH)

        if config.MODEL.TYPE == 'ResNet':
            self.__S__ = SpeakerNetModel(
                n_mels=config.DATA.N_MELS, 
                depths=config.MODEL.RESNET.DEPTHS, 
                dims=config.MODEL.RESNET.DIMS,
                embedding_size=config.MODEL.EMB_SIZE, 
                encoder_type=config.MODEL.AGGREGATION,
                specaug=specaug)

        elif config.MODEL.TYPE == 'TDNN':
            self.__S__ = SpeakerNetModel(
                n_mels=config.DATA.N_MELS,
                cnn_dim=config.MODEL.CNN_STEM.DIM, 
                ecapa_dim=config.MODEL.ECAPA.DIM, 
                embedding_size=config.MODEL.EMB_SIZE, 
                specaug=specaug)
        self.nPerSpeaker = config.DATA.N_PER_SPEAKER
        if torch.cuda.device_count() > 1:
            self.__S__ = nn.DataParallel(self.__S__)
        self.__S__ = self.__S__.to(device)

        ## Classifier
        LossFunction = importlib.import_module('loss.'+config.MODEL.LOSS.NAME).__getattribute__('LossFunction')
        self.__L__ = LossFunction(
            embedding_size=config.MODEL.EMB_SIZE,
            nClasses=config.MODEL.NUM_CLASSES,
            margin=config.MODEL.LOSS.MARGIN, 
            scale=config.MODEL.LOSS.SCALE,
            #total_iter=config.TRAIN.EPOCH_ITER*config.TRAIN.EPOCHS,
            n_per_speaker=config.DATA.N_PER_SPEAKER,
            total_iter=config.TRAIN.EPOCH_ITER * (config.TRAIN.EPOCHS - config.TRAIN.WARMUP_EPOCH)
        ).to(device)

        Optimizer = importlib.import_module('optimizer.'+config.TRAIN.OPTIMIZER.NAME).__getattribute__('Optimizer')
        self.optimizer = Optimizer(self.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=config.TRAIN.OPTIMIZER.SGD_MOMENTUM)
        # self.optimizer = Optimizer(self.parameters(), lr=0.01, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=config.TRAIN.OPTIMIZER.SGD_MOMENTUM)

        Scheduler = importlib.import_module('scheduler.'+config.TRAIN.SCHEDULER.NAME).__getattribute__('Scheduler')
        # self.scheduler = Scheduler(self.optimizer, step_size=config.TRAIN.SCHEDULER.STEPLR.STEP_SIZE, lr_decay=config.TRAIN.SCHEDULER.DECAY_RATE, last_epoch=config.TRAIN.START_EPOCH-1)
        # self.scheduler = Scheduler(self.optimizer, step_size=config.TRAIN.SCHEDULER.STEPLR.STEP_SIZE, lr_decay=config.TRAIN.SCHEDULER.STEPLR.DECAY_RATE)
        self.scheduler = Scheduler(self.optimizer, step_size=config.TRAIN.SCHEDULER.STEPLR.STEP_SIZE, lr_decay=config.TRAIN.SCHEDULER.STEPLR.DECAY_RATE, total_iter=config.TRAIN.EPOCH_ITER*(config.TRAIN.EPOCHS-config.TRAIN.WARMUP_EPOCH))

        self.device = device
        self.lr_step = config.TRAIN.SCHEDULER.STEPLR.LR_STEP
        self.emb_size = config.MODEL.EMB_SIZE
        self.init_lr = config.TRAIN.LR
        self.dul = config.MODEL.DUL
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.5f" % (
                sum(param.numel() for param in self.__S__.parameters()) / 1024 / 1024))
        assert self.lr_step in ['epoch', 'iteration']



    def train_network(self, loader, epoch, max_epochs, warmup_epoch, aug=False):
        self.train()
        epoch_iterator = len(loader)
        counter = 0
        loss    = 0
        top1    = 0    # EER or accuracy

        if epoch < warmup_epoch and epoch == 0: self.optimizer.param_groups[0]['lr'] = 0
        lr = self.optimizer.param_groups[0]['lr']

        print('epoch {}-th, lr is {}'.format(epoch, lr))

        loop = tqdm(loader, total=len(loader))

        for idx, (data, data_label) in enumerate(loop):

            data = data.transpose(1, 0)
            data = data.reshape(-1, data.size()[-1])
            # print(data.shape)
            if epoch < warmup_epoch: warm_up_lr(1 + epoch * epoch_iterator + idx, warmup_epoch * epoch_iterator,
                                                self.init_lr, self.optimizer)
            self.zero_grad()
            data = data.to(self.device)
            labels = data_label.to(self.device)
            # data_mos = data_mos.to(self.device)
            if self.dul:
               
                mu, logvar, speaker_feature = self.__S__.forward(data, aug=aug)

                nloss, prec = self.__L__.forward(mu=mu, logvar=logvar, x=speaker_feature, labels=labels,
                                                 ith_iter=epoch * epoch_iterator + idx + 1)
            
            else:
                speaker_feature = self.__S__.forward(data, aug=aug)
                nloss, prec = self.__L__.forward(x=speaker_feature, labels=labels,
                                                 ith_iter=epoch * epoch_iterator + idx + 1)
            nloss.backward()
            self.optimizer.step()
            loss += nloss.detach().cpu().item()
            top1 += prec.detach().cpu().item()
            counter += 1

            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(loss=loss / counter, acc=top1 / counter)

            if epoch >= warmup_epoch and self.lr_step == 'iteration': self.scheduler.step()

        if epoch >= warmup_epoch and self.lr_step == 'epoch': self.scheduler.step()
        
        return (loss/counter, top1/counter)

    def extract_full_segs_emb(self, filenames, max_frames=300, num_eval=10, test_data_path='your_dataset_path/vox1/wav'):
        """ full-length + 5 segments
        """
        self.eval()
        feats = []
        for filename in tqdm(filenames):
            filename = os.path.join(test_data_path, filename)
            audio1, audio2 = DataLoader.load_full_and_seg_wav(filename, max_frames=max_frames, num_eval=num_eval, eval_mode='seg')
            with torch.no_grad():
                data1 = torch.FloatTensor(audio1).to(self.device)
                data2 = torch.FloatTensor(audio2).to(self.device)
                # print('input data', data1.shape, data2.shape)
                emb1 = F.normalize(self.__S__(data1), p=2, dim=-1).cpu().numpy()
                emb2 = F.normalize(self.__S__(data2), p=2, dim=-1).mean(dim=-2).cpu().numpy()
                emb = (emb1 + emb2) / 2
                feats.append(emb)
        return feats

    def extract_full_emb(self, filenames, test_data_path='your_dataset_path/vox1/wav', **kwargs):
        self.eval()
        feats = []
        for filename in tqdm(filenames):
            filename = os.path.join(test_data_path, filename)
            audio, _ = DataLoader.load_full_and_seg_wav(filename, eval_mode='full')
            with torch.no_grad():
                data = torch.FloatTensor(audio).to(self.device)
                # print('input data', data1.shape, data2.shape)
                emb = F.normalize(self.__S__(data), p=2, dim=-1).cpu().numpy()
                feats.append(emb)
        return feats

    def extract_segs_emb(self, filenames=None, max_frames=300, num_eval=10, test_loader=None,
                         # test_data_path='your_dataset_path/vox1/wav'):
                         test_data_path='your_dataset_path//CN/eval'):
        self.eval()
        feats = []
        vars = []
        if test_loader is not None:
            for data in tqdm(test_loader):
                with torch.no_grad():
                    data = data.reshape(-1, 160 * max_frames + 240).to(self.device)
                    if self.dul:
                        o_embs = self.__S__(data)
                        embs = F.normalize(o_embs, p=2, dim=-1)
                        embs = embs.reshape(-1, num_eval, self.emb_size)
                        embs = embs.mean(dim=1)
                        feats.append(embs)
                    else:
                        o_embs = self.__S__(data)
                        embs = F.normalize(o_embs, p=2, dim=-1)
                        embs = embs.reshape(-1, num_eval, self.emb_size)
                        embs = embs.mean(dim=1)
                        feats.append(embs)

            feats = torch.concat(feats, dim=0)
            return feats


        for filename in tqdm(filenames):
            filename = os.path.join(test_data_path, filename)
            _, audio = DataLoader.load_full_and_seg_wav(filename, max_frames=max_frames, num_eval=num_eval,
                                                        eval_mode='full')
           
            with torch.no_grad():
                data = torch.FloatTensor(audio).to(self.device)
                emb = F.normalize(self.__S__(data), p=2, dim=-1).mean(dim=-2).cpu().numpy()
                feats.append(emb)
               
        return feats

    def extract_segs_emb_muti(self, filenames=None, max_frames=300, num_eval=10, test_loader=None,
                         test_data_path='your_dataset_path//CN/eval'):
        self.eval()
        feats = []
        if test_loader is not None:
            for data in tqdm(test_loader):
                with torch.no_grad():
                    data = data.reshape(-1, 160 * max_frames + 240).to(self.device)
                    if self.dul:
                        o_embs = self.__S__(data)
                        embs = F.normalize(o_embs, p=2, dim=-1)
                        embs = embs.reshape(-1, num_eval, self.emb_size)
                        embs = embs.mean(dim=1)
                        feats.append(embs)
                    else:
                        o_embs = self.__S__(data)
                        embs = F.normalize(o_embs, p=2, dim=-1)
                        embs = embs.reshape(-1, num_eval, self.emb_size)
                        embs = embs.mean(dim=1)
                        feats.append(embs)

            feats = torch.concat(feats, dim=0)
            return feats, None

        for filename in tqdm(filenames):
            filename = os.path.join(test_data_path, filename)
            _, audio = DataLoader.load_full_and_seg_wav(filename, max_frames=max_frames, num_eval=num_eval,
                                                        eval_mode='full')

            with torch.no_grad():
                data = torch.FloatTensor(audio).to(self.device)
                emb = F.normalize(self.__S__(data), p=2, dim=-1).mean(dim=-2).cpu().numpy()
                feats.append(emb)

        return feats

    def save_parameters(self, filename):

        opt_arr = filename.split('/')
        opt_arr[-1] = 'opt_' + opt_arr[-1]
        torch.save(self.optimizer.state_dict(), '/'.join(opt_arr))
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename):

        self_state = self.state_dict()
        # self.optimizer.load_state_dict(torch.load('/'.join(opt_arr)))
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.to(self.device)
        print("ok")
        print(filename)
        loaded_state = torch.load(filename)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    li = name.split('.')
                    name = '.'.join([li[0], 'module', *li[1:]])
                    if name not in self_state:
                        print("%s is not in the models." % origname)
                        continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, models: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
