import numpy as np
import scipy.signal
from math import ceil, floor, log2

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

    #label_format = "frame" or "time"
    def load(self, wav_file, phn_file, wrd_file, label_format="frame"):
        assert self._phn is not None, "Phoneme list is not loaded."
        assert self._wrd is not None, "Word list is not loaded."
        fs, data = self._loader.load(wav_file)
        N = data.shape[0]
        window_frame = int(self._mfcc_cording_params["winlen"] * fs)
        step_frame = int(self._mfcc_cording_params["winstep"] * fs)
        mfcc = self._mfcc_cord(fs, data, fft_size=window_frame)
        mfcc_d = delta(mfcc, self._delta_cording_param)
        mfcc_dd = delta(mfcc_d, self._delta_cording_param)
        M = mfcc.shape[0]
        if label_format == "time":
            window_frame = self._mfcc_cording_params["winlen"]
            step_frame = self._mfcc_cording_params["winstep"]
        phn = _label_cord(phn_file, self._phn, M, window_frame, step_frame)
        wrd = _label_cord(wrd_file, self._wrd, M, window_frame, step_frame)
        return ((mfcc, mfcc_d, mfcc_dd), phn, wrd)

    def _mfcc_cord(self, fs, data, fft_size=-1):
        kwargs  = Extractor._default_mfcc_cording_params.copy()
        winstep = kwargs["winstep"]
        winlen  = kwargs["winlen"]
        numcep  = kwargs["numcep"]
        preemph = kwargs["preemph"]
        ceplifter = kwargs["ceplifter"]

        data, nfft = split(data, fs, winlen, winstep, nfft=fft_size)
        data = applyPreEmphasis(data, preemph)
        data = applyWindow(data, np.hamming)
        spec = np.abs(np.fft.fft(data, nfft, axis=1))[:, :nfft//2]
        melfilterbank, _ = melFilterBank(fs, nfft, numcep+1)
        melspec = np.log10(np.dot(spec, melfilterbank.T))
        melceps = scipy.fftpack.realtransforms.dct(melspec, type=2, norm="ortho", axis=1)[:, 1:numcep+1]
        mfcc = liftering(melceps, ceplifter)
        return mfcc

def hz2mel(f):
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def applyPreEmphasis(signal, preemph):
    return scipy.signal.lfilter([1.0, -preemph], 1, signal, axis=1)

def applyWindow(signal, winfunc):
    window = winfunc(signal.shape[1])
    return window * signal

def melFilterBank(samplerate, nfft, numChannels):
    fmax = samplerate / 2
    melmax = hz2mel(fmax)
    nmax = nfft // 2
    df = samplerate / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c], dtype=int):
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c], dtype=int):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters

def liftering(cepstra, ceplifter):
    if ceplifter > 0:
        _, ncoeff = cepstra.shape
        n = np.arange(ncoeff)
        lift = 1 + (ceplifter/2.)*np.sin(np.pi*n/ceplifter)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def split(raw_data, samplerate, winlen, winstep, nfft=-1):
    win_frame = int(winlen * samplerate)
    step_frame = int(winstep * samplerate)
    if nfft < 0:
        nfft = 2 ** ceil(log2(win_frame))
        # nfft = win_frame
    assert win_frame <= nfft
    N = ceil((raw_data.shape[0] - win_frame) / step_frame) + 1
    raw_data = np.concatenate((raw_data, np.zeros((N-1)*step_frame+win_frame)), axis=0)
    indexies = np.arange(win_frame) + (np.arange(N) * step_frame).reshape((-1, 1)) + 1
    offset = nfft - win_frame
    left_padding = np.zeros((N, offset // 2))
    right_padding = np.zeros((N, offset - (offset // 2)))
    return np.concatenate((left_padding, raw_data[indexies], right_padding), axis=1), nfft

def delta(data, N):
    assert N >= 1
    NUMFRAMES = data.shape[0]
    delta = np.empty_like(data)
    # denominator = 2 * sum([i**2 for i in range(1, N+1)])
    denominator = N * (N+1) * (2*N+1) / 3.0
    padded = np.pad(data, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    v = np.arange(-N, N+1)
    for t in range(NUMFRAMES):
        delta[t] = np.dot(v, padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta
