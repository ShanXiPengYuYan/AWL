from __future__ import absolute_import
from __future__ import print_function

import glob
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import utils as ut
import DataLoader
from SpeakerNet import *
from metric import *
from config import get_config
# ===========================================
#        Parse the argument
# ===========================================

def parse_option():
    parser = argparse.ArgumentParser(description='Speaker Verification', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    
    parser.add_argument('--gpu', default='-1', type=str)
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--augment', type=bool)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--model', type=str, help="model name")
    parser.add_argument('--extract_embedding', type=str, default=None, help="extract_embedding")


    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    return args, config

def main():
    
    # ==================================
    #       Initialize.
    # ==================================
    args, config = parse_option()
    # GPU configureation
    ut.initialize_GPU(config.GPU)
    #---------------
    # mkdir checkpoint & log file path
    # ==================================
    #  init nisqa

    # ==================================
    model_params = [config.MODEL.NAME, config.MODEL.AGGREGATION]
    name = '_'.join(model_params)
    model_path, log_filename, board_path = ut.set_path(name)
    writer = SummaryWriter(board_path)
    ut.setup_seed(config.SEED)
    # ==================================
    #       Get Train dataset.
    # ==================================
    list_IDs, labels, num_classes = DataLoader.get_feat_label_numclass(train_sets=config.DATA.DATASET, vox1_path=config.DATA.VOX1_DATA_PATH, vox2_path=config.DATA.VOX2_DATA_PATH,
                                                                       cn1_path=config.DATA.CN1_DATA_PATH, cn2_path=config.DATA.CN2_DATA_PATH)

    spk_utt_info = DataLoader.get_spk_utt_info(labels)
    print(config.AUG.FLAG)
    dataset = DataLoader.trainDataset(list_IDs, labels, max_frames=config.DATA.MAX_FRAMES,
                                      augment=config.AUG.FLAG, musan_path=config.AUG.MUSAN_PATH, rir_path=config.AUG.RIR_PATH, spk_genre='vox2dev',
                                      )

    train_sampler = DataLoader.train_dataset_sampler(data_source=dataset, nPerSpeaker=config.DATA.N_PER_SPEAKER, max_seg_per_spk=config.DATA.MAX_SRG_PER_SPK,
                                                     batch_size=config.TRAIN.BATCH_SIZE, seed=config.SEED, distributed=False)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=DataLoader.worker_init_fn,
        sampler=train_sampler,
        drop_last=True,
    )
    train_sampler.__iter__()  # set num_samples
    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.TRAIN.EPOCH_ITER = len(train_loader)

    config.freeze()
    # ==================================
    #       Get Test dataset.
    # ==================================
    if config.TEST.MUTI:
        unique_list, utt2idx, enroll, test, verify_lb = DataLoader.get_test_utterance_list_muti(
            enroll_path=config.DATA.CN1_DATA_PATH, test_path=config.DATA.TEST_DATA_PATH)

    else:
        unique_list, utt2idx, genre2info  = DataLoader.get_test_utterance_list(test_sets=config.TEST.DATASET)
    test_loader = None
    if config.TEST.SPEED_UP:
        test_loader = DataLoader.get_test_loader(unique_list, test_data_path=config.DATA.TEST_DATA_PATH, pin_memory=config.DATA.PIN_MEMORY, num_workers=config.DATA.NUM_WORKERS, batch_size=config.TEST.BATCH_SIZE, max_frames=config.TEST.SEG.MAX_FRAMES, num_eval=config.TEST.SEG.NUM_EVAL)

    print('==> loaded test dataset')
    if config.TEST.AS_NORM.FLAG:
        print('===> apply AS_NORM, topK is %d'%config.TEST.AS_NORM)
    if config.TEST.MODE.endswith('seg'):
        print('===> seg max_frames is %d, num_eval is %d'%(config.TEST.SEG.MAX_FRAMES, config.TEST.SEG.NUM_EVAL))
    # ==================================
    #       Load spaker model.
    # ==================================
      # set num_sampler
    trainer = SpeakerNet(args, config)
    # ==> load pre-trained model
    if config.MODEL.RESUME:
        if os.path.isfile(config.MODEL.RESUME):
            trainer.load_parameters(config.MODEL.RESUME)
            print('==> successfully loading model {}.'.format(config.MODEL.RESUME))
        else:
            print("==> no checkpoint found at '{}'".format(config.MODEL.RESUME))

    print('==> gpu {} is, training {} images, classes: 0-{} loss: {}'.format(config.GPU, num_classes, num_classes - 1, config.MODEL.LOSS.NAME))

    # ==> model train
    best_eer = 99
    for epoch in range(config.TRAIN.START_EPOCH-1, config.TRAIN.EPOCHS):
        text = '%d epoch'%(config.TRAIN.START_EPOCH - 1)
        train_sampler.set_epoch(epoch)

        if epoch >= config.TRAIN.START_EPOCH:

            loss, acc = trainer.train_network(train_loader, epoch, config.TRAIN.EPOCHS, warmup_epoch=config.TRAIN.WARMUP_EPOCH, aug=config.AUG.FLAG)

            writer.add_scalar('loss', loss, epoch)
            writer.add_scalar('acc', acc, epoch)
            # loss, acc = 0, 1
            if config.SAVE:
                path = os.path.join(model_path, "weights-{:02d}.pt".format(epoch))
                trainer.save_parameters(path)
            # ==================================
            #       test VoxCeleb-O
            # ==================================
            text = '%d epoch, LOSS %.6f, ACC %.2f%%'%(epoch, loss, acc)

        if epoch < config.TRAIN.START_EPOCH or True: # or (epoch < 70 and epoch % 5 == 0) or epoch >= 70:

            if config.TEST.MODE == 'seg':
                embs = trainer.extract_segs_emb(unique_list, max_frames=config.TEST.SEG.MAX_FRAMES,
                                                num_eval=config.TEST.SEG.NUM_EVAL,
                                                test_loader=test_loader,
                                                test_data_path=config.DATA.VOX1_DATA_PATH)
            elif config.TEST.MODE == 'full':
                pass
            elif config.TEST.MODE == 'full_seg':
                pass

            as_norm = None
            if config.TEST.AS_NORM.FLAG:
                as_norm = {
                    'imposters_emb': load_imposter_embs(path),
                    'topk': config.TEST.AS_NORM.TOPK,
                }

            for key, (list1, list2, verify_lb) in genre2info.items():
                eer, th, dcf1, dcf2 = evaluate(list1, list2, verify_lb, utt2idx, embs, as_norm)
                best_eer = min(eer, best_eer)
                print('Best error rate is {:6f}%'.format(best_eer * 100), end=' ')
                print('Equal error rate is {:6f}%, at threshold {:6f}'.format(
                    eer * 100, th), end=' ')
                print('Minimum detection cost (0.01) is {:6f}'.format(dcf1), end=' ')
                print('Minimum detection cost (0.05) is {:6f}'.format(dcf2))
            writer.add_scalar('EER', eer, epoch)
            writer.add_scalar('minDCF_0.01', dcf1, epoch)
            writer.add_scalar('minDCF_0.05', dcf2, epoch)
            writer.add_scalar('EER', eer, epoch)
            text += ', EER {:2f}%, bestEER {:2f}%, minDCF0.01 {:6f}'.format(eer*100, best_eer*100, dcf1)
        train_sampler.__iter__()
        with open(log_filename, 'a+') as f:
            f.write(text+'\n')

if __name__ == "__main__":
    main()
