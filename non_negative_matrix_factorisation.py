"""
This is the NMF class which includes all the required libraries and 
functions which are used with different initializations such as NMF with
random initialization, NMF harmonic and NMF score informed.
"""

from __future__ import division
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

class NMF():
    def __init__(self):
        self.gray_values = 1 - (np.geomspace(start=0.1, stop=1.1, num=512) - 0.1)
        self.gray_values_rgb = np.repeat(self.gray_values.reshape(512, 1), 3, axis=1)
        self.color_wb = LinearSegmentedColormap.from_list('color_wb', self.gray_values_rgb, N=512)

    def find_kl_divergence(self, x, y):
        """
        Find KL divergence between given input x and y.
        Input :
            X and Y are numpy arrays of the same dimension
        Output :
            Returns the KL divergence value between given inputs
        """
        assert(x.shape==y.shape)
        initial_mask = np.logical_and(x > 0, y > 0)
        initial_value = x[initial_mask] * (np.log(x[initial_mask] / np.float32(y[initial_mask]))) - x[initial_mask] + y[initial_mask]

        second_mask = np.logical_and(np.logical_and(np.logical_not(initial_mask), x == 0), y > 0)
        second_value = y[second_mask]

        final_mask = np.logical_and(np.logical_not(initial_mask), np.logical_not(second_mask))
        final_value = np.zeros((final_mask.sum(), ))

        return np.mean(np.concatenate((initial_value, second_value, final_value)))

    # @jit('float64(float64[:,:], float64[:,:], float64[:,:])')
    # def matmul_numba(self, mat_1, mat_2, result_mat):
    #     """
    #     Used to multiply input matrices using JIT compiler
    #     """
    #     for i in range(len(mat_1)):
    #         for j in range(len(mat_2[0])):
    #             for k in range(len(mat_2)):
    #                 result_mat[i][j] += mat_1[i][k] * mat_2 [k][j]
    #     return result_mat

    def nmf_(self, V, R, eps = 0.001, use_iteration_check = False, iteration_count = 5000, W = None, H = None, distance_measure = 'default', view_error = True, use_numba = False):
        """
        Non negative matrix factorisation of a given input matrix into matrices of smaller rank
        using specified distance measure. Eucledian distance measure is used by default.
        Input arguments:
            V : Non negative matrix of size KxN
            R : Rank parameter
            eps : Threshold epsilon
            use_iteration_check : Boolean if program is terminated after iteration count
            iteration_count : Number of iterations
            W : Matrix of dimension KxR for initialisation
            H : Matrix of dimension RxN for initialisation
            distance_measure : Type of distance metric to be used for training NMF
                                'default' : Eucledian distance metric
                                'kl_div' : KL divergence metric
                                Supported 'ord' values for np.linalg.norm
            view_error : Displays error values for W and H matrices
            use_numba : Boolean if numba speed up is used or not
        Output arguments:
            W : Non negative matrix W of dimension KxR
            H : Non negative matrix H of dimension RxN
            V_approx : Approximated version of V
            residual_error : Error between V and V_approx
            
        """
        (K, N) = V.shape
        print("Input shape of V is (K x N) : ", K, N)
        W_temp = np.zeros((K, R), dtype=np.float64)
        H_temp = np.zeros((R,N), dtype=np.float64)
        if W is None:
            np.random.seed(0)
            W = np.random.rand(K,R)
            print("W matrix initialised as random.")
        if H is None:
            np.random.seed(0)
            H = np.random.rand(R,N)
            print("H matrix initialised as random.")
        W_ret = W
        eps_test_H = np.inf
        eps_test_W = np.inf
        iter_ = 0
        run = True
        if use_numba:
            nb.jit(nopython = True)
            print("Numba speed-up used")
        else:
            print("Numba speed-up not selected")
        while(run):
            print('Iteration : ', iter_)
            H_temp = np.multiply(H, np.nan_to_num(  np.divide(np.dot(np.transpose(W),V),
                                                    np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps)
                                                )
                                )
            W_temp = np.multiply(W, np.nan_to_num(  np.divide(np.dot(V,np.transpose(H_temp)), 
                                                    np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps)
                                                )
                                )
            if distance_measure == 'default':     
                eps_test_H = np.linalg.norm(H-H_temp)
                eps_test_W = np.linalg.norm(W-W_temp)
            elif distance_measure == 'kl_div':
                eps_test_H = find_kl_divergence(H-H_temp)
                eps_test_W = find_kl_divergence(W-W_temp)
            else:
                try:
                    eps_test_H = np.linalg.norm(H-H_temp, ord=distance_measure)
                    eps_test_W = np.linalg.norm(W-W_temp, ord=distance_measure)
                except:
                    warnings.warn('Chosen distance measure not supported, using default method')
                    eps_test_H = np.linalg.norm(H-H_temp)
                    eps_test_W = np.linalg.norm(W-W_temp)
            H = H_temp
            W = W_temp
            iter_ = iter_ + 1
            if (eps_test_H < eps) and (eps_test_W < eps):
                run = False
                #A manual check can be placed which breaks the loop if iter count is large
            if(iter_ > iteration_count) and use_iteration_check:
                print('Number of iterations achieved.')
                break
        if view_error:
            print("Error in H is :", eps_test_H)
            print("Error in W is :", eps_test_W)
        V_approx = W.dot(H)
        residual_error = np.linalg.norm(V-V_approx, ord=2)    
    
        return W, H, V_approx, residual_error

    def xlabel_generator_for_plots (self, x):
        """
        Generates x axis labels for plots as per the type of matrices
        """
        x_label = { 'V_input': 'Time (s)', 'W_learnt': 'components', 'H_learnt': 'Time (s)', 'W_initial': 'components', 'H_initial': 'Time (s)', 
                    'V_approx': 'Time (s)'}
        return x_label[x]

    def ylabel_generator_for_plots (self, x):
        """
        Generates y axis labels for plots as per the type of matrices
        """
        y_label = { 'V_input': 'Freq (Hz)', 'W_learnt': 'Freq (Hz)', 'H_learnt': 'components', 'W_initial': 'Freq (Hz)', 'H_initial': 'components', 
                    'V_approx': 'Freq (Hz)'}
        return y_label[x]

    def title_generator_for_plots (self, x):
        """
        Generates title for plots as per the type of matrices
        """
        title = { 'V_input': 'Orig Spectrogram', 'W_learnt': 'Learnt W', 'H_learnt': 'Learnt H', 'W_initial': 'Initial W', 
                    'H_initial': 'Initial H', 'V_approx': 'Approx Spectrogram'}
        return title[x]

    def generate_plots_of_matrices(self, V_input, W_learnt, H_learnt, W_initial = None, H_initial = None, V_approx = None, generate_subplots = True,
                                    figure_size = (12,15), save_figures = False, **kwargs):
        """
        Generates matplotlib images of the matrices
        Input:
            V_input : Input Non negative matrix which as to be approximated of dimension KxN
            W_learnt : Learnt Matrix W of dimension KxR 
            H_learnt : Learnt Matrix H of dimension RxN
            W_initial : Initialisation matrix for W matrix
            H_initial : Initialisation matrix for H
            V_approx : Approximation of V_input
            generate_subplots : Boolean to specify if subplots are required, W and H matrices are grouped,
                                V and V_approx are grouped together
            figure_size : Tuple specifying the figure size
            save_figures : Boolean to specify if the plot images have to be saved. Default names chosen as 
                            per the matrix type when in non subplot matrices. In subplot mode, file name
                            is 'matrix_subplot'.png
            **kwargs : Supliment arguments for specifying vmax, cmap, sampling frequency, hop length
        Output:
            Matplotlib figures of the specified matrices either in subplots or separately
        """
        vmax = kwargs.pop('vmax', 0.5)
        cmap = kwargs.pop('vmax', 'Blues')
        Fs = kwargs.pop('sampling_frequency_Fs', 0)
        hop_length = kwargs.pop('hop_length', 0)
        R = kwargs.pop('R', 8)
        plt.figure(figsize=(figure_size))
        matrices = [V_input, W_learnt, H_learnt, W_initial, H_initial, V_approx]
  
        matrix_names = ['V_input', 'W_learnt', 'H_learnt', 'W_initial', 'H_initial', 'V_approx']
        if generate_subplots :
            for i in range(1,7):
                plt.subplot(3,2,i)
                if matrix_names[i-1] == 'W_learnt' or matrix_names[i-1] == 'W_initial':
                    plt.xticks(np.arange(0.5, R + 0.5, 1), np.arange(0, R, 1))
                    try:
                        librosa.display.specshow(matrices[i-1], y_axis='linear', x_axis='frames', vmax=vmax, cmap=cmap, sr=Fs, hop_length=hop_length)  
                    except AttributeError:
                        librosa.display.specshow(np.zeros( matrices[1].shape ), y_axis='linear', x_axis='frames', vmax=vmax, cmap=cmap, sr=Fs, hop_length=hop_length)  
                else:
                    try:
                        librosa.display.specshow(matrices[i-1], y_axis='frames', x_axis='time', vmax=vmax, cmap=cmap, sr=Fs, hop_length=hop_length)  
                    except AttributeError:
                        librosa.display.specshow(np.zeros( matrices[2].shape ), y_axis='frames', x_axis='time', vmax=vmax, cmap=cmap, sr=Fs, hop_length=hop_length)  
                
                if matrix_names[i-1] == 'H_learnt' or matrix_names[i-1] == 'H_initial':
                    plt.yticks(np.arange(0.5, R+0.5,1), np.arange(0,R,1))
                plt.xlabel(self.xlabel_generator_for_plots(matrix_names[i-1]))
                plt.ylabel(self.ylabel_generator_for_plots(matrix_names[i-1]))
                plt.title(self.title_generator_for_plots(matrix_names[i-1]))
                plt.tight_layout()
                if W_initial == None and H_initial == None:
                    plt.suptitle('Random Initialisation')
                elif W_initial != None and H_initial == None:
                    plt.suptitle('Template Initialisation')
                else :
                    plt.suptitle('Template Initialisation')
            plt.show()
            if save_figures :
                plt.savefig('NMF matrices.png', bbox_inches = 'tight')

    def find_pitches_from_annotation(self, annotation_file):
        """
            Find pitches from input annotation file in CSV format.
            Files used are of either of two formats:
                Type 1 -> Time Stamp (s) ; Pitch
                Type 2 -> Track ID, ticks ; Note on/off ; idx ; pitch ; velocity
            Returns a set of unique pitches
        """
        annotations = pandas.read_csv(annotation_file)
        pitches_ = annotations['pitch']
        return np.unique(pitches_)

    def template_pitch(self, number_of_frequency_points = 25, pitch = 64, frequency_resolution = 0, tolerance = 0.05):
        """
        Generates templates for a given pitch. By default 25 harmonics of A4 are calculated.
    
        number_of_frequency_points : Number of harmonics to be generated
        pitch : Fundamental pitch
        frequency_resolution : Frequency resolution = Fs/hop_length
        tolerance : Relative frequency tolerance for the harmonics
    
        Returns: 
            Non-negative list of templates of size (number_of_frequency_points, )
        """    

        ### check if this concept to be used or if my concept for matrix direct.
        ## Here, vector is generated and then matrix ? How is final size taken? Maybe use STFT size as per mine?
        max_freq = number_of_frequency_points * frequency_resolution
        pitch_freq = 2**((pitch - 69) / 12) * 440
        max_order = int(np.ceil(max_freq / ((1 - tolerance) * pitch_freq)))
        template = np.zeros(number_of_frequency_points)
        for m in range(1, max_order + 1):
            min_idx = max(0, int((1- tolerance) * m * pitch_freq / frequency_resolution))
            max_idx = min(number_of_frequency_points-1, int((1 + tolerance) * m * pitch_freq / frequency_resolution))
            template[min_idx:max_idx+1] = 1/m
            return template 

    # W_prime2 = np.zeros((W_2.shape[0], W_2.shape[1]))
    # T_coef = np.arange(X2.shape[1]) * H2 / Fs2
    # F_coef = np.arange(X2.shape[0]) * Fs2 / N2
    # left = min(T_coef)
    # right = max(T_coef) + N2 / Fs2
    # lower = min(F_coef)
    # upper = max(F_coef)
    # ratio_ = (upper/np.float(X2.shape[0]))

    # #freqUsed = [207.65, 196.00, 130.81, 155.56, 415.30, 493.88, 261.63, 311.13, 392.00, 932.33, 523.25, 587.33, 622.25, 783.99]
    # freqUsed = [77.7817, 155.563, 65.4064, 130.813, 195.998, 207.652, 391.995, 466.164, 311.127, 783.991, 523.251, 622.254, 587.330, 415.305, 830.609]
    # for i in range(W_prime2.shape[1]):
    #     temp1 = freqUsed[i]
    #     for p in range(1,25):
    #         temp_cord = np.int(np.around((temp1*p)/ratio_))
    #         W_prime2[temp_cord-2:temp_cord+(2*p), i] = 1 #10/(np.float(p)) #try inverting the size and inverting values
       


    def dupLists(self, input_list):
        """
        Find duplicate elements within a given list 
        Input:
            input_list : Python list
        Output:
            duplicates : List of duplicate items
            uniques : List of unique items
        """
        elements = set()  
        duplicates = set()  
        elements_add = elements.add 
        duplicates_add = duplicates.add
        for item in L:
            if item in elements:
                duplicates_add(item)
            else:
                elements_add(item)
        return list(duplicates), list(elements)