import numpy as np
from python_speech_features import mfcc, delta
from math import ceil, floor

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
    raw_label = np.loadtxt(f, dtype=[('col1', 'f16'), ('col2', 'f16'), ('col3', 'S10')])
    return raw_label

def _label_cord(label_file, label_list, length, window_frame, step_frame, init_val=0):
    label_ary = np.ones(length, dtype=int) * init_val
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
        "numcep": 13,
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

    #label_format = "frame" or "time"
    def load(self, wav_file, phn_file, wrd_file, label_format="frame"):
        assert self._phn is not None, "Phoneme list is not loaded."
        assert self._wrd is not None, "Word list is not loaded."
        assert label_format in ("frame", "time")
        fs, data = self._loader.load(wav_file)
        N = data.shape[0]
        mfcc = self._mfcc_cord(data, fs)
        mfcc_d = delta(mfcc, self._delta_cording_param)
        mfcc_dd = delta(mfcc_d, self._delta_cording_param)
        M = mfcc.shape[0]
        if label_format == "time":
            window_len = self._mfcc_cording_params["winlen"]
            step_len = self._mfcc_cording_params["winstep"]
        phn = _label_cord(phn_file, self._phn, M, window_len, step_len)
        wrd = _label_cord(wrd_file, self._wrd, M, window_len, step_len)
        return (mfcc, mfcc_d, mfcc_dd), phn, wrd

    def _mfcc_cord(self, data, fs):
        kwargs = self._mfcc_cording_params
        fft_size = int(kwargs["winlen"] * fs)
        return mfcc(data, samplerate=fs, nfft=fft_size, **kwargs)[:, 1:]