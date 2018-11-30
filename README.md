# SpeechFeatureExtraction
-----
## Author information
Author: Ryo Ozaki<br>
E-mail: ryo.ozaki@em.ci.ritsumei.ac.jp

## How to use

1. Install
  ```
  git clone https://github.com/RyoOzaki/SpeechFeatureExtraction
  cd SpeechFeatureExtraction
  python setup.py install
  ```
2. Write code

  1. Import<br>
    ```
    from speech_feature_extraction import Extractor
    from speech_feature_extraction.util import WavLoader, SPHLoader
    ```

  2. Make an instance of Extractor with specifying WavLoader or SPHLoader
    ```
    extractor = Extractor(<Loader>, phn_list=<List of phoneme labels>, wrd_list=<List of word labels>)
    ```
    or
    ```
    extractor = Extractor(WavLoader)
    extractor.load_phoneme_list(<List of phoneme label files>)
    extractor.load_word_list(<List of word label files>)
    ```
    If you want to set the specific index to some labels, please "sp" argument.<br>
    E.g.
    ```
    extractor.load_phoneme_list(****, sp={0: "sil"})
    ```
    In this case, the label "sil" is inserted to index 0.

  3. Extract features
    ```
    extractor.load(<Wave file>, <Phoneme label file>, <Word label file>)
    ```
    If the label is written by based time, please set the argument "label_format" to "time".
    ```
    extractor.load(****, label_format="time")
    ```
    The function "load" returns MFCC, &Delta;MFCC, &Delta;&Delta;MFCC, phoneme labels, and word labels.
    ```
    (mfcc, d_mfcc, dd_mfcc), p_lab, w_lab = extractor.load(****)
    ```

---------
## Additional information
  - Default parameters of MFCC extracting<br>
    - winlen - the length of the analysis window in seconds. Default value is 0.025 [sec].
    - winstep - the step between successive windows in seconds. Default value is 0.01 [sec].
    - numcep - the number of cepstrum to return, default 12.
    - nfilt - the number of filters in the filterbank, default 20.
    - preemph - apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    - ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    - winfunc - the analysis window to apply to each frame. Default is numpy.hamming.
  - If you want to change the parameters of MFCC extracting<br>
    Please set the argument when making Extractor instance.<br>
    E.g.
    ```
    extractor = Extractor(****, numcep=13)
    ```
