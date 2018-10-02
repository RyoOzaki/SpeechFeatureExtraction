import numpy as np
from math import floor, ceil, log2
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import delta as psf_delta

def _load_label_list(files, sp=None):
    s = set()
    for f in files:
        raw_label = _load_raw_label(f)
        s |= set([raw[2] for raw in raw_label])
    l = list(s)
    l.sort()
    if sp is not None and type(sp) is dict:
        s -= set(sp.values())
        l = list(s)
        l.sort()
        for k in sorted(sp.keys()):
            l.insert(k, sp[k])
    return l

def _load_raw_label(f):
    raw_label = np.loadtxt(f, dtype=[('col1', 'i8'), ('col2', 'i8'), ('col3', 'S10')])
    # begin = [raw[0] for raw in raw_label]
    # end   = [raw[1] for raw in raw_label]
    # label = [raw[2] for raw in raw_label]
    # return (begin, end, label)
    return raw_label

def _label_cord(label_file, label_list, length, window_frame, step_frame, init_val=0):
    label_ary = np.ones(length) * init_val
    raw_labels = _load_raw_label(label_file)
    for b,e,l in raw_labels:
        left = int(max(0, floor((2*b-window_frame)/(2*step_frame))))
        right = int(ceil((2*e-window_frame)/(2*step_frame)))
        label_ary[left:right] = label_list.index(l)
    return label_ary

class Extractor(object):

    _default_mfcc_cording_params={
        "winlen": 25.0 / 1000,
        "winstep": 10.0 / 1000,
        "numcep": 12,
        "nfilt": 20,
        "preemph": 0.97,
        "ceplifter": 22,
        "appendEnergy": False,
        "winfunc": np.hamming
    }

    def __init__(self, wav_loader, phn_list=None, wrd_list=None, delta_cording_param=2, **kwargs):
        self._mfcc_cording_params = dict(Extractor._default_mfcc_cording_params.copy(), **kwargs)
        self._delta_cording_param = delta_cording_param
        self._loader = wav_loader
        self._phn  = phn_list
        self._wrd  = wrd_list

    @property
    def phoneme_list(self):
        return self._phn

    @property
    def word_list(self):
        return self._wrd

    def load_phoneme_list(self, phn_files, **kwargs):
        self._phn = _load_label_list(phn_files, **kwargs)
        return self._phn

    def load_word_list(self, wrd_files, **kwargs):
        self._wrd = _load_label_list(wrd_files, **kwargs)
        return self._wrd

    def load(self, wav_file, phn_file, wrd_file):
        if self._phn is None:
            raise RuntimeError("Phoneme list is not loaded.")
        if self._wrd is None:
            raise RuntimeError("Word list is not loaded.")
        fs, data = self._loader.load(wav_file)
        N = data.shape[0]
        window_frame = int(self._mfcc_cording_params["winlen"] * fs)
        step_frame = int(self._mfcc_cording_params["winstep"] * fs)
        fft_size = 2 ** ceil(log2(N))
        mfcc = self._mfcc_cord(fs, data, fft_size)
        mfcc_d = psf_delta(mfcc, self._delta_cording_param)
        mfcc_dd = psf_delta(mfcc_d, self._delta_cording_param)
        M = mfcc.shape[0]
        phn = _label_cord(phn_file, self._phn, M, window_frame, step_frame)
        wrd = _label_cord(wrd_file, self._wrd, M, window_frame, step_frame)
        return ((mfcc, mfcc_d, mfcc_dd), phn, wrd)


    def _mfcc_cord(self, fs, data, fft_size):
        return psf_mfcc(data, samplerate=fs, nfft=fft_size, **self._mfcc_cording_params)
