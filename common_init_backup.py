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

#log color scheme
# gray_values = 1 - (np.geomspace(0.001, 1.001, 256) - 0.001)
gray_values = 1 - (np.geomspace(start=0.1, stop=1.1, num=512) - 0.1)
gray_values_rgb = np.repeat(gray_values.reshape(512, 1), 3, axis=1)
color_wb = LinearSegmentedColormap.from_list('color_wb', gray_values_rgb, N=512)


#NMF with random initialization
def NMF(V, R, eps):
    (K, N) = V.shape
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    W = np.random.rand(K,R)
    W_ret = W
    H = np.random.rand(R,N)
    eps_test_H = np.inf
    eps_test_W = np.inf
    iter_ = 0
    run = True
    while(run):
        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.divide(temp_1,temp_2))
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.divide(temp_3, temp_4))


        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
            #A manual check can be placed which breaks the loop if iter count is large
        if(iter_ > 5000):
            print('broken')
            print(eps_test_H)
            print(eps_test_W)
            break

    print("Error in H is :", eps_test_H)
    print("Error in W is :", eps_test_W)
    print("Number of interations :", iter_)
    return W, H, W_ret



#NMF with harmonically initialised W and random H

def NMF_const(V, R, eps, W_prime):
    (K, N) = V.shape
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    W = np.reshape(W_prime, (K,R))
    np.random.seed(0)
    H = np.random.rand(R,N)
    eps_test_H = np.inf
    eps_test_W = np.inf
    H_dist = []
    W_dist = []
    iter_ = 0
    run = True
    while(run):
        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.nan_to_num(np.divide(temp_1,temp_2)))
        #H_temp = np.nan_to_num(H_temp)
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.nan_to_num(np.divide(temp_3, temp_4)))
        W_temp = np.nan_to_num(W_temp)

        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H_dist = np.append(H_dist, eps_test_H)
        W_dist = np.append(W_dist, eps_test_W)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
#         if(iter_ > 4000):
#             print('broken')
#             print(eps_test_W, eps_test_H)
#             break

    print("Error in H is :", eps_test_W)
    print("Error in W is :", eps_test_H)
    print("Number of interations :", iter_)
    return W, H


#Score initialised NMF

def NMF_const_score(V, R, eps, W_prime, temp_H_score):
    (K, N) = V.shape
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    #W = np.reshape(W_prime, (K,R))
    W = W_prime
    H = temp_H_score
    eps_test_H = np.inf
#     eps_test_H = 10
    eps_test_W = np.inf
    H_dist = []
    W_dist = []
    iter_ = 0
    run = True
    while(run):
        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.nan_to_num(np.divide(temp_1,temp_2)))
        H_temp = np.nan_to_num(H_temp)
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.nan_to_num(np.divide(temp_3, temp_4)))
        W_temp = np.nan_to_num(W_temp)

        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H_dist = np.append(H_dist, eps_test_H)
        W_dist = np.append(W_dist, eps_test_W)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
        if(iter_ > 4000):
            print('broken')
            print(eps_test_W, eps_test_H)
            break

    print("Error in H is :", eps_test_W)
    print("Error in W is :", eps_test_H)
    print("Number of interations :", iter_)
    return W, H


#function to find the KL divergence distance between x and y
def my_kl_div3(x, y):

    mask1 = np.logical_and(x > 0, y > 0)
    val1 = x[mask1] * (np.log(x[mask1] / np.float32(y[mask1]))) - x[mask1] + y[mask1]

    mask2 = np.logical_and(np.logical_and(np.logical_not(mask1), x == 0), y > 0)
    val2 = y[mask2]

    mask3 = np.logical_and(np.logical_not(mask1), np.logical_not(mask2))
    val3 = np.zeros((mask3.sum(), ))

    return np.mean(np.concatenate((val1, val2, val3)))


@jit('float64(float64[:,:], float64[:,:], float64[:,:])')
def matmul_numba(mat1, mat2, rmat):
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                rmat[i][j] += mat1[i][k] * mat2 [k][j]
    return rmat


#Numba based random NMF
def nu_NMF(V, R, eps):
    (K, N) = V.shape
    np.random.seed(0)
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    W = np.random.rand(K,R)
    W_ret = W
    H = np.random.rand(R,N)
    eps_test_H = np.inf
    eps_test_W = np.inf
    iter_ = 0
    run = True
    nb.jit(nopython = True)
    while(run):
        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.divide(temp_1,temp_2))
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.divide(temp_3, temp_4))
        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
            #A manual check can be placed which breaks the loop if iter count is large
        if(iter_ > 5000):
            print('broken')
            print(eps_test)
            break

    print("Error in H is :", eps_test_H)
    print("Error in W is :", eps_test_W)
    print("Number of interations :", iter_)
    return W, H

#Numba based template constraint NMF
def nu_NMF_const(V, R, eps, W_prime):
    (K, N) = V.shape
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    W = np.reshape(W_prime, (K,R))
    H = np.random.rand(R,N)
    eps_test_H = np.inf
    eps_test_W = np.inf
    H_dist = []
    W_dist = []
    iter_ = 0
    run = True
    nb.jit(nopython=True)
    while(run):

        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.nan_to_num(np.divide(temp_1,temp_2)))
        H_temp = np.nan_to_num(H_temp)
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.nan_to_num(np.divide(temp_3, temp_4)))
        W_temp = np.nan_to_num(W_temp)

        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H_dist = np.append(H_dist, eps_test_H)
        W_dist = np.append(W_dist, eps_test_W)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
#         if(iter_ > 4000):
#             print('broken')
#             print(eps_test_W, eps_test_H)
#             break

    print("Error in H is :", eps_test_W)
    print("Error in W is :", eps_test_H)
    print("Number of interations :", iter_)
    return W, H

#To find duplicates and unique elements in a list
def dupLists(L):
    elements = set()  #returns an empty set
    duplicates = set()  #returns an empty set
    elements_add = elements.add  #add elements into the set
    duplicates_add = duplicates.add
    for item in L:
        if item in elements:
            duplicates_add(item)
        else:
            elements_add(item)
    return list(elements), list(duplicates)

#Numba based score constraint NMF
def nu_NMF_const_score(V, R, eps, W_prime, temp_H_score):
    (K, N) = V.shape
    print("Shape of V is (K x N) : ", K, N)
    W_temp = np.zeros((K, R), dtype=np.float64)
    H_temp = np.zeros((R,N), dtype=np.float64)
    W = np.reshape(W_prime, (K,R))
    H = temp_H_score
    eps_test_H = np.inf
    #eps_test_H = 10
    eps_test_W = np.inf
    H_dist = []
    W_dist = []
    iter_ = 0
    run = True
    nb.jit(nopython=True)
    while(run):

        temp_1 = np.dot(np.transpose(W),V)
        temp_2 = np.dot(np.dot(np.transpose(W),W), H) + np.finfo(float).eps
        H_temp = np.multiply(H, np.nan_to_num(np.divide(temp_1,temp_2)))
        H_temp = np.nan_to_num(H_temp)
        temp_3 = np.dot(V,np.transpose(H_temp))
        temp_4 = np.dot(np.dot(W,H_temp),np.transpose(H_temp)) + np.finfo(float).eps
        W_temp = np.multiply(W, np.nan_to_num(np.divide(temp_3, temp_4)))
        W_temp = np.nan_to_num(W_temp)

        eps_test_H = np.linalg.norm(H-H_temp)
        eps_test_W = np.linalg.norm(W-W_temp)
        H_dist = np.append(H_dist, eps_test_H)
        W_dist = np.append(W_dist, eps_test_W)
        H = H_temp
        W = W_temp
        iter_ = iter_ + 1
        if (eps_test_H < eps) and (eps_test_W < eps):
            run = False
        if(iter_ > 4000):
            print('broken')
            print(eps_test_W, eps_test_H)
            break

    print("Error in H is :", eps_test_W)
    print("Error in W is :", eps_test_H)
    print("Number of interations :", iter_)
    return W, H
