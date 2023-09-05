import numpy as np
import torch
import torch.nn.functional as F
from operator import itemgetter

# calculation of two evaluation indices.
def compute_fnr_fpr(scores, labels):
    """ computes false negative rate (FNR) and false positive rate (FPR)
    given trial scores and their labels.
    """

    indices = np.argsort(scores)
    labels = labels[indices]

    target = (labels == 1).astype('f8')
    nontar = (labels == 0).astype('f8')

    fnr = np.cumsum(target) / np.sum(target)
    fpr = 1 - np.cumsum(nontar) / np.sum(nontar)
    print(fnr.shape)
    # with open('./drawDCT/Triple.txt', 'w') as f:
    #     f.write(str(fnr.tolist()))
    #     f.write('\n')
    #     f.write(str(fpr.tolist()))

    return fnr, fpr

def compute_eer(fnr, fpr, requires_threshold=False, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
        *kaldi style*
    """

    diff_miss_fa = fnr - fpr
    x = np.flatnonzero(diff_miss_fa >= 0)[0]
    eer = fnr[x - 1]
    if requires_threshold:
        assert scores is not None
        scores = np.sort(scores)
        th = scores[x]
        return eer, th
    return eer

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return c_det / c_def

# score normalization.
def load_imposter_embs(filename):
    """ load imposter from model's weight.
    """
    load_dict = torch.load(filename)
    # print(load_dict.get('__L__.weight', None))
    imposters_embs = load_dict.get('__L__.weight', None)
    print(imposters_embs.shape)
    assert isinstance(imposters_embs, torch.Tensor), 'Not exist Loss weights in file {}.'.format(filename)
    imposter_embs = F.normalize(imposters_embs, p=2, dim=1)
    return imposter_embs

def get_zero_score(enroll, test, imposter_embs, topK=-1, eps=1e-12):
    # v1, v2 shape is (1, dim)
    verify_cosine = torch.squeeze(enroll @ test.T).numpy()
    imposter_cosines = torch.squeeze(enroll @ imposter_embs.T)
    if topK > 0: imposter_cosines = imposter_cosines.topk(topK).values
    mean = imposter_cosines.mean().numpy()
    std = imposter_cosines.std().numpy()
    score = (verify_cosine - mean) / (std + eps)
    return score

def get_test_score(enroll, test, imposter_embs, topK=-1, eps=1e-12):
    return get_zero_score(test, enroll, imposter_embs, topK=topK, eps=eps)

def get_symmetric_score(enroll, test, imposter_embs, topK=-1, eps=1e-12):
    zero_score = get_zero_score(enroll, test, imposter_embs, topK=topK, eps=1e-12)
    test_score = get_test_score(enroll, test, imposter_embs, topK=topK, eps=1e-12)
    score = (zero_score+test_score) / 2
    return score

# Given score and label, to calculate EER and minDCF
def calculate_score(enrolls, tests, utt2idx, embs, as_norm=None):
    scores = []
    for enroll, test in zip(enrolls, tests):
        enroll_idx, test_idx = utt2idx[enroll], utt2idx[test]
        enroll_emb, test_emb = embs[enroll_idx], embs[test_idx]
        if as_norm and isinstance(as_norm, dict):
            imposter_embs = as_norm.get('imposter_embs')
            topK = as_norm.get('topK', -1)
            score = get_symmetric_score(enroll_emb, test_emb, imposter_embs, topK)
        else:
            score = (enroll_emb @ test_emb.T).squeeze()
        
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
        scores.append(score)
    
    scores = np.array(scores)
    return scores

def calculate_score_muti(enrolls, tests, utt2idx, embs, as_norm=None, qualitys=None):
    scores = np.array([])
    enroll_emb_dict = {}
    batch_size = 256
    enroll_emb_li = []
    test_emb_li = []
    for enroll, test in zip(enrolls, tests):
        test_idx = utt2idx[test]  # idx is (start, number)
        if type(embs) == np.ndarray:
            test_emb = embs[test_idx[0]: test_idx[0] + test_idx[1], :]
            if enroll in enroll_emb_dict:
                enroll_emb = enroll_emb_dict.get(enroll)
            else:
                enroll_idx = utt2idx[enroll]
                enroll_emb = embs[enroll_idx[0]: enroll_idx[0] + enroll_idx[1], :]
                enroll_emb = enroll_emb.mean(axis=0, keepdims=True)
                enroll_emb_dict[enroll] = enroll_emb
        else:
            test_emb = embs[test_idx[0]: test_idx[0] + test_idx[1], :]
            # torch.from_numpy(test_emb)
            if enroll in enroll_emb_dict:
                enroll_emb = enroll_emb_dict.get(enroll)
            else:
                enroll_idx = utt2idx[enroll]
                enroll_emb = embs[enroll_idx[0]: enroll_idx[0] + enroll_idx[1], :]
                if qualitys != None:
                    enroll_qualitys = qualitys[enroll_idx[0]: enroll_idx[0] + enroll_idx[1], :]
                # changed
                if qualitys == None:
                    # enroll_emb = enroll_emb.mean(dim=0, keepdim=True)
                    # raw
                    qualitys = torch.norm(enroll_emb, p=2, dim=-1, keepdim=True)
                    enroll_emb = enroll_emb.mean(dim=0, keepdim=True)
                    enroll_emb = F.normalize(enroll_emb, p=2, dim=-1)

                else:
                    qualitys = torch.norm(enroll_emb, p=2, dim=-1, keepdim=True)
                    # enroll_emb = (torch.exp(qualitys) / (torch.exp(qualitys).sum(dim=0, keepdim=True))) * enroll_emb
                    enroll_emb = enroll_emb.mean(dim=0, keepdim=True)
                    enroll_emb = F.normalize(enroll_emb, p=2, dim=-1)




                enroll_emb_dict[enroll] = enroll_emb

        enroll_emb_li.append(enroll_emb)
        test_emb_li.append(test_emb)
        if len(enroll_emb_li) == batch_size:
            score = get_seg_score(enroll_emb_li, test_emb_li, as_norm)
            enroll_emb_li, test_emb_li = [], []
            scores = np.concatenate([scores, score])

    if len(enroll_emb_li) > 0:
        score = get_seg_score(enroll_emb_li, test_emb_li, as_norm)
        del enroll_emb_li, test_emb_li
        scores = np.concatenate([scores, score])

    return scores

def get_seg_score(enroll_emb_li, test_emb_li, as_norm=None):
    enroll_embs = torch.cat(enroll_emb_li)
    test_embs = torch.cat(test_emb_li)
    # enroll_embs = torch.from_numpy(np.concatenate(enroll_emb_li))
    # test_embs = torch.from_numpy(np.concatenate(test_emb_li))
    if as_norm and isinstance(as_norm, dict):
        imposter_embs = as_norm.get('imposter_embs')
        topK = as_norm.get('topK', -1)
        score = get_symmetric_score(enroll_embs, test_embs, imposter_embs, topK)
    else:
        score = F.cosine_similarity(enroll_embs, test_embs).squeeze()

    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()

    return score

def evaluate(enrolls, tests, verify_lb, utt2idx, embs, as_norm=None):
    # print(embs.shape)
    """the describtion of evalute.
    Params:
        enrolls: consists of enroll lists such as ../../*.wav.
        tests: consists of test lists such as ../../*.wav.
        verify_lb: enroll vs test label.
        utt2idx: utterance correspond to index of embs.
        feats: all utterances' embeddings.
        as_norm: is a dict and consists of imposter cohorts' embedings and topK, such as:
        {
            'imposter_embs': *, // shape: $ N \times embs $
            'topK': .
        }.
    Return:
        eer, threhold, minDCF_0.01, minDCF_0.001.
    """
    scores = calculate_score(enrolls, tests, utt2idx, embs, as_norm=as_norm)
    fnr, fpr = compute_fnr_fpr(scores, verify_lb)
    eer, th = compute_eer(fnr, fpr, True, scores)
    return eer, th, compute_c_norm(fnr, fpr, 0.01), compute_c_norm(fnr, fpr, 0.05)

def evaluate_muti(enrolls, tests, verify_lb, utt2idx, embs, as_norm=None, qualitys=None):
    """the describtion of evalute.
    Params:
        enrolls: consists of enroll lists such as ../../*.wav.
        tests: consists of test lists such as ../../*.wav.
        verify_lb: enroll vs test label.
        utt2idx: utterance correspond to index of embs.
        feats: all utterances' embeddings.
        as_norm: is a dict and consists of imposter cohorts' embedings and topK, such as:
        {
            'imposter_embs': *, // shape: $ N \times embs $
            'topK': .
        }.
    Return:
        eer, threhold, minDCF_0.01, minDCF_0.001.
    """
    scores = calculate_score_muti(enrolls, tests, utt2idx, embs, as_norm=as_norm, qualitys=qualitys)
    fnrs, fprs, thresholds = ComputeErrorRates(scores, verify_lb)
    eer, eer_th = ComputeEER(scores, verify_lb)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
    return eer, mindcf  # , compute_c_norm(fnr, fpr, 0.001)

class Indicate:
    def __init__(self, eer=99., minDCF=1.) -> None:
        self.eer = eer
        self.minDCF = minDCF

    def set_best_minDCF(self, eer, minDCF):
        if self.minDCF > minDCF:
            self.minDCF = minDCF
            self.eer = eer
        if self.minDCF == minDCF:
            self.eer = min(self.eer, eer)

    def set_best_eer(self, eer, minDCF):
        if self.eer > eer:
            self.eer = eer
            self.minDCF = minDCF
        if self.eer == eer:
            self.minDCF = min(self.minDCF, minDCF)

def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def ComputeEER(scores, labels):
    target_scores = []
    nontarget_scores = []
    for x in zip(scores, labels):
        if (x[1] == 1):
            target_scores.append(x[0])
        else:
            nontarget_scores.append(x[0])
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_position = 0
    for i in range(target_size - 1):
        target_position = i
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    threshold = target_scores[target_position]
    # print ("threshold is --> ",threshold)
    eer = target_position * 1.0 / target_size
    return eer, threshold


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold