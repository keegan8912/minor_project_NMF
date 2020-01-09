#from non_negative_matrix_factorisation import *
#from playsound import playsound
from __future__ import division
import non_negative_matrix_factorisation
from non_negative_matrix_factorisation import NMF
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import librosa
import os
import time
import scipy.signal as sig
import scipy
import librosa.display
import soundfile as sf
from IPython.display import Audio
from matplotlib.colors import LinearSegmentedColormap
import numba as nb
from numba import jit
import itertools
import pandas
import pretty_midi
import collections
import warnings
import pickle


def nmf_for_audio_signal(audio_, window_size, hop_size, window_type, audio_preview = False, verbose = True, use_spec = False, **kwargs):
    """
    Performs nmf for audio input
    Input :
        audio_ : Path for audio signal in mp3 or wav format
        window_size : Window size for STFT
        hop_size : STFT hop size
        window_type : Type of window used in STFT
                        'hanning' or 'default' : Hanning window
                        'hamming' : Hamming window
        audio_preview : Boolean to play the audio input
        verbose : Boolean for printing additional information
        use_spec : Use spectrogram of audio signal as non negative matrix
        **kwargs : 
            Arguments for member function nmf of class NMF
            V, R, eps, use_iteration_check, iteration_count, W, H, distance_measure, view_error, use_numba

    Output :
        Outputs of function nmf from class NMF
    """
    input_x, sampling_frequency_Fs = sf.read(audio_)
    if audio_preview :
        try:
            playsound(audio_)
        except:
            warnings.warn('Audio format may not be supported, skipping playback.')
    window = np.hanning(window_size)
    input_stft_X = librosa.core.stft(input_x, n_fft = window_size, hop_length = hop_size, window = window)
    input_spectrogram_Y = np.abs(input_stft_X)**2
    if verbose:
        print("Shape of STFT is :", input_stft_X.shape)
        print("Shape of Spectrogram is :", input_spectrogram_Y.shape)
    R = kwargs.pop('R', 8)
    eps = kwargs.pop('eps', 0.001)
    use_iteration_check = kwargs.pop('use_iteration_check', False)
    iteration_count = kwargs.pop('iteration_count', 5000)
    W = kwargs.pop('W', None)
    H = kwargs.pop('H', None)
    distance_measure = kwargs.pop('distance_measure', 'default')
    view_error = kwargs.pop('view_error', True)
    use_numba = kwargs.pop('use_numba', False)

    t = time.time()
    if use_spec:
        V = input_spectrogram_Y
    else:
        V = np.abs(input_stft_X)
    
    obj = NMF()
    with open('stft_objects.pkl', 'wb') as f:
        pickle.dump([window_size, hop_size, sampling_frequency_Fs], f)
    W, H, V_approx, residual_error = obj.nmf_(V, R, eps, use_iteration_check = use_iteration_check, iteration_count = iteration_count, 
                                             W = W, H = H, distance_measure = distance_measure, view_error = view_error, use_numba = use_numba
                                             )
    elapsed = time.time()-t
    if verbose:
        print("Shape of Learnt W : ", W.shape)
        print("Shape of Learnt H : ", H.shape)
        print("Time elapsed (sec) : ", elapsed)
    return V, W, H, V_approx, residual_error


#needed for plotting only
# T_coef = np.arange(X.shape[1]) * H / Fs
# F_coef = np.arange(X.shape[0]) * Fs / N
# left = min(T_coef)
# right = max(T_coef) + N / Fs
# lower = min(F_coef)
# upper = max(F_coef)
# duration = (len(x)//Fs) + 1
# ratio_ = (upper/X.shape[0])