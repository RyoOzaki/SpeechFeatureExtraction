
import scipy.io.wavfile as wav
class WavLoader(object):
    @classmethod
    def load(cls, f):
        (rate,data) = wav.read(f)
        if data.ndim == 2:
            # data = data.mean(axis=0)
            data = data[:,0]
            # data = data[:,1]
        return (rate, data)

from sphfile import SPHFile
class SPHLoader(object):
    @classmethod
    def load(cls, f):
        sph = SPHFile(f)
        return (sph.format['sample_rate'], sph.content)
