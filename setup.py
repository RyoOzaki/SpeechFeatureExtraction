from setuptools import setup, find_packages

setup(
    name='SpeechFeatureExtraction',
    version='0.0.2',
    description='Package of the speech feature extraction',
    author='Ryo Ozaki',
    author_email='ryo.ozaki@em.ci.ritsumei.ac.jp',
    url='https://github.com/RyoOzaki/SparseAutoencoder',
    license=license,
    install_requires=['numpy', 'scipy', 'python-speech-features', 'sphfile'],
    packages=['speech_feature_extraction', 'speech_feature_extraction.util']
)
