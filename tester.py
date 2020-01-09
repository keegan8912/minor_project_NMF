from __future__ import division
from non_negative_matrix_factorisation import NMF
from music_nmf import nmf_for_audio_signal
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



obj = NMF()
# x = np.random.rand(1,10)
# y = np.random.rand(1,10)

# pitches_found = find_pitches_from_annotation('backend/chopin_L_tx1.csv')
# pitches_found = find_pitches_from_annotation('backend/chopinR.csv')
# print('Shape of pitches : ', pitches_found.shape)

pitches = [130.813, 146.8325, 164.814, 174.614, 195.995, 220, 246.941, 261.625]

def make_template_matrice_from_pitches(frequency_resolution, tolerance, number_of_frequency_points, pitches = [64], annotation_file = None):
    if pitches == None :
        pitches_found = obj.find_pitches_from_annotation(annotation_file)
    else :
        pitches_found = pitches
    Rank_R = len(pitches_found)
    template_matrix = np.zeros((number_of_frequency_points, Rank_R))
    for rank in range(Rank_R):
        template_matrix[:, rank] = obj.template_pitch(number_of_frequency_points = number_of_frequency_points, pitch = pitches_found[rank], frequency_resolution = frequency_resolution, tolerance = 0.05)
    return template_matrix

# KL_div_x_y = obj.find_kl_divergence(x,y)
input_x, sampling_frequency_Fs = sf.read('backend/FMP_C2_F10.wav')
Fs = 22050
hop_length = 512
frequency_resolution = sampling_frequency_Fs/hop_length
template_matrix = make_template_matrice_from_pitches(frequency_resolution = frequency_resolution, tolerance = 0.05, number_of_frequency_points = 25, pitches = pitches)
print('Template matrix shape : ', template_matrix.shape)


V_input, W_learnt, H_learnt, V_approx, residual_error = nmf_for_audio_signal('backend/FMP_C2_F10.wav', window_size=4096, hop_size=512, window_type='hanning', 
                                                    audio_preview=False, verbose=True, use_spec=False, R=8, eps=1, iteration_count=10
                                                    )


with open('stft_objects.pkl', 'rb') as f:
    window_size, hop_size, sampling_frequency_Fs = pickle.load(f)
 

obj.generate_plots_of_matrices(V_input, W_learnt, H_learnt, W_initial = None, H_initial = None, V_approx = np.dot(W_learnt, H_learnt), 
                                generate_subplots = True, figure_size = (15,5), save_figures = False, 
                                sampling_frequency_Fs = sampling_frequency_Fs, hop_size = hop_size)



# #For pitches find say 25 templates, pitch 64, find 25 harmonics, say 8 pitches, find templates for each pitch.
# #matrix is then 25x8 ; len of pitches should be rank.
