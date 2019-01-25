import numpy as np
from python_speech_features import mfcc, delta
from python_speech_features.sigproc import preemphasis, framesig, powspec, logpowspec
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

    _default_cording_params={
        "winlen": 25.0 / 1000,
        "winstep": 10.0 / 1000,
        "numcep": 12,
        "nfilt": 20,
        "preemph": 0.97,
        "ceplifter": 22,
        "appendEnergy": False,
        "winfunc": np.hamming,
        "highfreq": None
    }

    def __init__(self, wav_loader, phn_list=None, wrd_list=None, delta_cording_param=2, **kwargs):
        self._cording_params = dict(Extractor._default_cording_params.copy(), **kwargs)
        self._cording_params["numcep"] += 1
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
        mfcc = self._mfcc_cord(data, fs)
        mfcc_d = delta(mfcc, self._delta_cording_param)
        mfcc_dd = delta(mfcc_d, self._delta_cording_param)
        if label_format == "time":
            window_len = self._cording_params["winlen"]
            step_len = self._cording_params["winstep"]
        else:
            window_len = int(self._cording_params["winlen"] * fs)
            step_len = int(self._cording_params["winstep"] * fs)
        M = mfcc.shape[0]
        phn = _label_cord(phn_file, self._phn, M, window_len, step_len)
        wrd = _label_cord(wrd_file, self._wrd, M, window_len, step_len)
        return (mfcc, mfcc_d, mfcc_dd), phn, wrd

    def load_without_label(self, wav_file):
        fs, data = self._loader.load(wav_file)
        mfcc = self._mfcc_cord(data, fs)
        mfcc_d = delta(mfcc, self._delta_cording_param)
        mfcc_dd = delta(mfcc_d, self._delta_cording_param)
        return mfcc, mfcc_d, mfcc_dd

    def load_powspec(self, wav_file, phn_file, wrd_file, label_format="frame"):
        assert self._phn is not None, "Phoneme list is not loaded."
        assert self._wrd is not None, "Word list is not loaded."
        assert label_format in ("frame", "time")
        fs, data = self._loader.load(wav_file)
        pspec = self._powspec_cord(data, fs)
        if label_format == "time":
            window_len = self._cording_params["winlen"]
            step_len = self._cording_params["winstep"]
        else:
            window_len = int(self._cording_params["winlen"] * fs)
            step_len = int(self._cording_params["winstep"] * fs)
        M = pspec.shape[0]
        phn = _label_cord(phn_file, self._phn, M, window_len, step_len)
        wrd = _label_cord(wrd_file, self._wrd, M, window_len, step_len)
        return pspec, phn, wrd

    def load_logpowspec(self, wav_file, phn_file, wrd_file, label_format="frame"):
        assert self._phn is not None, "Phoneme list is not loaded."
        assert self._wrd is not None, "Word list is not loaded."
        assert label_format in ("frame", "time")
        fs, data = self._loader.load(wav_file)
        pspec = self._logpowspec_cord(data, fs)
        if label_format == "time":
            window_len = self._cording_params["winlen"]
            step_len = self._cording_params["winstep"]
        else:
            window_len = int(self._cording_params["winlen"] * fs)
            step_len = int(self._cording_params["winstep"] * fs)
        M = pspec.shape[0]
        phn = _label_cord(phn_file, self._phn, M, window_len, step_len)
        wrd = _label_cord(wrd_file, self._wrd, M, window_len, step_len)
        return pspec, phn, wrd

    def load_powspec_without_label(self, wav_file):
        fs, data = self._loader.load(wav_file)
        pspec = self._powspec_cord(data, fs)
        return pspec

    def load_logpowspec_without_label(self, wav_file):
        fs, data = self._loader.load(wav_file)
        pspec = self._logpowspec_cord(data, fs)
        return pspec

    def _mfcc_cord(self, data, fs):
        kwargs = self._cording_params
        fft_size = int(kwargs["winlen"] * fs)
        return mfcc(data, samplerate=fs, nfft=fft_size, **kwargs)[:, 1:]

    def _powspec_cord(self, data, fs):
        kwargs = self._cording_params
        preemph = kwargs["preemph"]
        winlen = kwargs["winlen"]
        winstep = kwargs["winstep"]
        winfunc = kwargs["winfunc"]
        fft_size = int(winlen * fs)
        data = preemphasis(data, preemph)
        frames = framesig(data, winlen * fs, winstep * fs, winfunc)
        return powspec(frames, fft_size)

    def _logpowspec_cord(self, data, fs):
        kwargs = self._cording_params
        preemph = kwargs["preemph"]
        winlen = kwargs["winlen"]
        winstep = kwargs["winstep"]
        winfunc = kwargs["winfunc"]
        fft_size = int(winlen * fs)
        data = preemphasis(data, preemph)
        frames = framesig(data, winlen * fs, winstep * fs, winfunc)
        ps = powspec(frames, fft_size)
        return np.log(ps + 1.0)
        # return logpowspec(frames, fft_size, norm=False)
