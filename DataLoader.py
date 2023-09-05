import imp
import os
import json
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, BatchSampler
import soundfile as sf
from scipy import signal
import utils as ut
import torch.distributed as dist
import librosa as lb

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_spk_utt_info(labels):

    spk_utt_info = {}
    pre_label = labels[0]
    pre_idx = 0
    for idx, label in enumerate(labels):
        if pre_label != label:
            spk_utt_info[pre_label] = (pre_idx, idx-1)
            pre_idx = idx
            pre_label = label

    spk_utt_info[pre_label] = (pre_idx, idx)
    
    return spk_utt_info


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames, spk_genre='vox2dev'):
        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15], 'spk':[13, 20]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1], 'spk':[3,7]}
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'));

        for file in augment_files:

            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        # self.noiselist['spk'] = get_spk_noise_list(spk_genre)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))


    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio, sr  = load_random_wav(noise, self.max_frames, 0)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = sf.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

def load_full_and_seg_wav(filename, samplerate=None, max_frames=300, num_eval=10, eval_mode='full_seg'):
    assert max_frames > 0, "max_frames don't less 0 in load segment wav"
    # Read wav file and convert to torch tensor

    audio, sample_rate = sf.read(filename)
    # print(audio)
    # Maximum audio length
    max_audio = max_frames * 160 + 240
    audiosize = audio.shape[0]
    feat1, feat2 = None, None
    if eval_mode.startswith('full'):

        feats = [audio]
        feat1 = np.stack(feats,axis=0).astype(np.float)

    if eval_mode.endswith('seg'):
        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = np.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]

        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
        
        feats = []
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feat2 = np.stack(feats,axis=0).astype(np.float)

    return feat1, feat2


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = sf.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat

def load_random_wav(filename, max_frames=300, fill_value=None, samplerate=None):
    if max_frames == -1:
        audio, samplerate = sf.read(filename, samplerate=samplerate)
    else:
        max_audio = max_frames * 160 + 240
        with sf.SoundFile(filename, 'r', samplerate) as f:
            audio_size = f.frames
            start = 0 if max_audio > audio_size else np.int64(np.random.random()*(audio_size-max_audio))
            frames = f._prepare_read(start, None, max_audio)
            audio = f.read(frames, 'float64', False, fill_value)
            if max_audio > audio_size and fill_value is None:
                audio = np.pad(audio, (0, max_audio - audio_size), "wrap")

            samplerate = f.samplerate
    audio = np.expand_dims(audio, 0)
    return audio, samplerate

def get_feat_label_numclass(train_sets=['vox1dev', 'vox2test', 'vox2dev', 'cn1train', 'cn2train'],
                            vox1_path='your_dataset_path/vox1/wav', vox2_path='your_dataset_path/vox2/wav',
                            cn1_path='/data/chenjunyu/data/CN/cn-celeb1', cn2_path='/data/chenjunyu/data/CN/cn-celeb2'):
    """ genre [vox1dev, vox2test, vox2dev]
    """
    max_idx = 0
    list_IDs, labels = [], []
    for train_set in train_sets:
        if train_set == 'vox1dev':
            strings = np.loadtxt('./meta/voxlb1_dev.txt', str)
        elif train_set == 'vox1test':
            strings = np.loadtxt('./meta/voxlb1_test.txt', str)
        elif train_set == 'vox2dev':
            strings = np.loadtxt('./meta/voxlb2_dev.txt', str)
        elif train_set == 'vox2test':
            strings = np.loadtxt('./meta/voxlb2_test.txt', str)
        else:
            raise Warning("[Warning] train_sets exist Invalid data")
        
        if train_set.startswith('vox1'):
            list_IDs.extend([os.path.join(vox1_path, string[0]) for string in strings])
        elif train_set.startswith('vox2'):
            list_IDs.extend([os.path.join(vox2_path, string[0]) for string in strings])

        labels.extend([max_idx + int(string[1].strip()) for string in strings])
        max_idx = 1 + max(labels)

    return list_IDs, labels, max_idx


def get_feat_label_numclass_nisqa(train_sets=['vox1dev', 'vox2test', 'vox2dev', 'cn1train', 'cn2train'],
                            vox1_path='your_dataset_path/vox1/wav',
                            vox2_path='your_dataset_path/vox2/wav',
                            ):
    """ genre [vox1dev, vox2test, vox2dev]
    """
    max_idx = 0
    list_IDs, labels, mos = [], [], []
    for train_set in train_sets:
        if train_set == 'vox1dev':
            strings = np.loadtxt('./meta/voxlb1_dev_nisqa.txt', str)
        elif train_set == 'vox1test':
            strings = np.loadtxt('./meta/voxlb1_test.txt', str)
        elif train_set == 'vox2dev':
            strings = np.loadtxt('./meta/voxlb2_dev.txt', str)
        elif train_set == 'vox2test':
            strings = np.loadtxt('./meta/voxlb2_test.txt', str)
        else:
            raise Warning("[Warning] train_sets exist Invalid data")

        if train_set.startswith('vox1'):
            list_IDs.extend([os.path.join(vox1_path, string[0]) for string in strings])
        elif train_set.startswith('vox2'):
            list_IDs.extend([os.path.join(vox2_path, string[0]) for string in strings])
        elif train_set.startswith('cn1'):
            list_IDs.extend([os.path.join(cn1_path, string[0]) for string in strings])
        elif train_set.startswith('cn2'):
            list_IDs.extend([os.path.join(cn2_path, string[0]) for string in strings])

        labels.extend([max_idx + int(string[1].strip()) for string in strings])
        mos.extend([float(string[2].strip()) for string in strings])
        max_idx = 1 + max(labels)

    return list_IDs, labels, mos, max_idx

def get_test_utterance_list(test_sets=['O', 'H', 'E','SITW'], cleared=True):
    total_list = None
    genre2info = {}
    for test_set in test_sets:
        if test_set == 'O':
            verify_list = np.loadtxt('./meta/voxceleb1_veri_test_fixed.txt', str)
        elif test_set == 'H':
            verify_list = np.loadtxt('./meta/voxceleb1_veri_test_hard_fixed.txt', str)
        elif test_set == 'E':
            verify_list = np.loadtxt('./meta/voxceleb1_veri_test_extended_fixed.txt', str)
        elif test_set == 'SITW':
            verify_list = np.loadtxt('./meta/sitw_veri_test.txt', str)
        else:
            raise ValueError("Not exist")

        verify_lb = np.array([int(i[0]) for i in verify_list])
        list1 = np.array([i[1] for i in verify_list])
        list2 = np.array([i[2] for i in verify_list])

        genre2info[test_set] = [ list1, list2, verify_lb ]
        total_list = np.concatenate((list1, list2)) if total_list is None else np.concatenate((total_list, list1, list2))

    unique_list = np.unique(total_list)
    utt2idx = {}
    for idx, utt in enumerate(unique_list):
        utt2idx[utt] = idx

    return unique_list, utt2idx, genre2info


class trainDataset(Dataset):
    
    def __init__(self, list_IDs, labels, max_frames, augment, musan_path, rir_path, spk_genre,**kwargs):

                 # mos,# nisqa
        print('trainDataset load, max_frames is {}, augment is {}'.format(max_frames, augment))
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames, spk_genre=spk_genre)
        self.list_IDs = list_IDs
        self.labels = labels
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment
        # self.mos = mos

    def __getitem__(self, indices):
        feat = []
        for index in indices:

            audio, _ = load_random_wav(self.list_IDs[index], self.max_frames, fill_value=None)
            # audio = loadWAV(self.list_IDs[index], self.max_frames, evalmode=False)

            if self.augment:
                # augtype = random.randint(0, 5)
                augtype = random.randint(0, 8)
                audio = self.augment_by_type(audio=audio, augtype=augtype)
            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.labels[index]

    def __len__(self):
        return len(self.list_IDs)

    def augment_by_type(self, audio, augtype):
        if augtype == 0 or augtype == 1 or augtype == 2  or augtype == 8:  # Original
            audio = audio
        elif augtype == 3:  # Reverberation
            audio = self.augment_wav.reverberate(audio)
        elif augtype == 4:  # Babble
            audio = self.augment_wav.additive_noise('speech', audio)
        elif augtype == 5:  # Music
            audio = self.augment_wav.additive_noise('music', audio)

        return audio

    def get_librosa_melspec(
            self,
            feat,
            sr=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=64,
            fmax=16e3,
            ms_channel=None,
    ):
        '''
        Calculate mel-spectrograms with Librosa.
        '''
        # Calc spec

        sr = 16000
        # hop_length = int(sr * hop_length)
        # win_length = int(sr * win_length)

        S = lb.feature.melspectrogram(
            y=feat,
            sr=sr,
            S=None,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=1.0,

            n_mels=n_mels,
            fmin=0.0,
            fmax=fmax,
            htk=False,
            norm='slaney',
        )

        spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
        return spec

class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label = data_source.labels
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        # self.num_samples = 68480
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()
        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)
        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):

            data = data_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)
            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # print(flattened_label)
        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []
        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = round_down(len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            # print(self.num_samples)
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_num_samples(self) -> int:
        return self.num_samples


class RepeatedSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, utt_num=4):
        super().__init__(sampler, batch_size, drop_last)
        self.utt_num = utt_num

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for _ in range(self.utt_num):
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size * self.utt_num  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size * self.utt_num  # type: ignore[arg-type]


def get_train_loader(list_IDs, labels, batch_size, num_workers, max_frames=300, augment=False, musan_path='your_dataset_path/musan_split', rir_path='your_dataset_path/RIRS_NOISES/simulated_rirs',
                     seed=0, spk_genre='vox2dev',nPerSpeaker=2, max_seg_per_spk=1002, distributed=False,  **kwargs):

    dataset = trainDataset(list_IDs, labels, max_frames, augment, musan_path, rir_path, spk_genre)
    train_sampler = train_dataset_sampler(data_source=dataset, nPerSpeaker=nPerSpeaker, max_seg_per_spk=max_seg_per_spk,
                                          batch_size=batch_size, seed=seed, distributed=distributed)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        # generator=g,
        sampler=train_sampler,
        # shuffle=True,
        drop_last=True,
    )
    return train_sampler, train_loader

class testDataset(Dataset):
    def __init__(self, list_IDs, max_frames=400, num_eval=10, test_data_path='your_dataset_path/vox1/wav') -> None:
        super().__init__()
        self.list_IDs = list_IDs
        self.max_frames = max_frames
        self.num_eval = num_eval
        self.test_data_path = test_data_path
        assert max_frames != -1, 'max_frames is not equal -1.'
    
    def __getitem__(self, index):
        filename = os.path.join(self.test_data_path, self.list_IDs[index])
        _, feat = load_full_and_seg_wav(filename, max_frames=self.max_frames, num_eval=self.num_eval, eval_mode='seg')
        return torch.FloatTensor(feat)

    def __len__(self):
        return len(self.list_IDs)

def get_test_loader(unique_list, test_data_path, pin_memory, num_workers, batch_size=1, max_frames=400, num_eval=10, **kwargs):
    dataset = testDataset(unique_list, max_frames, num_eval, test_data_path)
    test_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory)
    return test_loader

# Hard Sample TODO
class tuneDataset(Dataset):
    pass

class tuneSampler():
    pass

class get_tuneTrain_loader():
    pass

if __name__ == '__main__':
    pass