import os
from complex_matrix import *
import numpy as np
import torch
import math
import torch.nn as nn
import random
import time
import datetime
import scipy as sp
from scipy import integrate
from array import array
######
SIGMA = 1
def Guss(x):
    u = 0 
    sigma = SIGMA  
    y_sig = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    return y_sig

def xGuss(x):
    return x*Guss(x)

def uniform(x):
    return 3/np.pi/2

def xuniform(x):
    return 3*x/np.pi/2

def Lloyd_Max(N_level, minlevel, maxlevel, flag):
    mapping_table = np.zeros(N_level)
    bound_table = np.zeros(N_level+1)
    bound_table[0] = minlevel
    bound_table[N_level] = maxlevel
    bound_table[1:N_level] = np.random.uniform(minlevel, maxlevel, N_level-1)
    bound_table = np.sort(bound_table)
    for i in range(200):
        if flag == 'Gaussian':  
            for k in range(N_level):
                t1,err1 = integrate.quad(xGuss, bound_table[k], bound_table[k+1])
                t2,err1 = integrate.quad(Guss, bound_table[k], bound_table[k+1])
                mapping_table[k]= t1/t2
            for k in range(N_level-1):
                bound_table[k+1] = 0.5*(mapping_table[k]+mapping_table[k+1])
        elif flag == 'Uniform':    
            for k in range(N_level):
                t1,err1 = integrate.quad(xuniform, bound_table[k], bound_table[k+1])
                t2,err1 = integrate.quad(uniform, bound_table[k], bound_table[k+1])
                mapping_table[k]= t1/t2
            for k in range(N_level-1):
                bound_table[k+1] = 0.5*(mapping_table[k]+mapping_table[k+1])
    return mapping_table, bound_table


def LloydQuan(bound_table, mapping_table, input=None, N_level=16):
    ########  mapping process
    if input<bound_table[1]:
        return mapping_table[0]
    elif input>=bound_table[N_level-1]:
        return mapping_table[N_level-1]
    for i in range(1,N_level-1):
        if input>=bound_table[i] and input<bound_table[i+1]:
            return mapping_table[i]


Nt = 16         # T antennas
Nr = 16        # R antennas
NtRF = 4        # RF chains at the transmitter
NrRF = 4        # RF chains at the receiver
Ncl = 2         # clusters
Nray = 4        # ray
sigma_h = 0.3   # gain
Tao = 0.001     # delay
fd = 3          # maximum Doppler shift


quan_bits =  2

minlevel_G = -3*SIGMA   # Three times of the standard sigma
maxlevel_G = 3*SIGMA

minlevel_U = -np.pi/3
maxlevel_U = np.pi/3
N_level = 2**quan_bits

mapping_table_G, bound_table_G = Lloyd_Max(N_level, minlevel_G, maxlevel_G, flag='Gaussian')

N_level = 2**quan_bits
mapping_table_U, bound_table_U = Lloyd_Max(N_level, minlevel_U, maxlevel_U, flag='Uniform')

def theta(N, phi, Batch_size):
    a = np.zeros((Batch_size, Ncl*Nray, N, 1), dtype=complex)
    f = 0
    for i in range(Batch_size):
        for j in range(Ncl*Nray):
            f += 1
            for z in range(N):
                a[i][j][z] = np.exp(1j * np.pi * z * np.sin(phi[f-1]))
    PHI = phi.reshape(Batch_size, Ncl*Nray)
    return a/np.sqrt(N), PHI

def H_gen_mmWave(Batch_size):
    HH = torch.zeros((Batch_size, 2, Nr, Nt))
    alpha_h = np.random.normal(0, sigma_h, (Batch_size, Ncl*Nray)) + 1j*np.random.normal(0, sigma_h, (Batch_size,Ncl*Nray))
    
    phi_ar = np.zeros(Batch_size*Ncl*Nray)  
    phi_at = np.zeros(Batch_size*Ncl*Nray)  
    for i in range(Batch_size*Ncl*Nray):
        phi_ar[i] = np.random.uniform(-np.pi/3, np.pi/3)
        phi_at[i] = np.random.uniform(-np.pi/3, np.pi/3)

    ar, ThetaR = theta(Nr, phi_ar, Batch_size)
    at, ThetaT = theta(Nt, phi_at, Batch_size)
    for b in range(Batch_size):
        H = np.zeros((Nr, Nt), dtype=complex)
        fff = 0
        for i in range(Ncl):
            for j in range(Nray):
                H += alpha_h[b][fff] * np.dot(ar[b][fff], np.conjugate(at[b][fff]).T)
                fff += 1
        H = np.sqrt(Nt * Nr / Ncl * Nray) * H
        H = c2m(H)
        H = H.to(dtype=torch.float)
        HH[b] = H
    # HH = HH.numpy()

    alpha_h_quan = torch.zeros(Batch_size, 2, Ncl*Nray)
    alpha_h_quan[:,0,:] = torch.tensor(np.real(alpha_h))
    alpha_h_quan[:,1,:] = torch.tensor(np.imag(alpha_h))
    length = Batch_size*Ncl*Nray*2
    alpha_h_quan = alpha_h_quan.view(length)
    for i in range(length):
        alpha_h_quan[i] = LloydQuan(bound_table_G, mapping_table_G, alpha_h_quan[i], N_level=N_level)
    alpha_h_quan = alpha_h_quan.view(Batch_size, 2, Ncl*Nray)
    alpha_h_quan = alpha_h_quan[:,0,:].numpy() + 1j*alpha_h_quan[:,1,:].numpy()


    phi_ar_quan = np.zeros(Batch_size*Ncl*Nray)  
    phi_at_quan = np.zeros(Batch_size*Ncl*Nray)  
    for i in range(Batch_size*Ncl*Nray):
        phi_ar_quan[i] = LloydQuan(bound_table_U, mapping_table_U, phi_ar[i], N_level=N_level)
        phi_at_quan[i] = LloydQuan(bound_table_U, mapping_table_U, phi_at[i], N_level=N_level)

    ar_quan, ThetaR_quan = theta(Nr, phi_ar_quan, Batch_size)
    at_quan, ThetaT_quan = theta(Nt, phi_at_quan, Batch_size)

    HH_quan = torch.zeros((Batch_size, 2, Nr, Nt))
    for b in range(Batch_size):
        H_quan = np.zeros((Nr, Nt), dtype=complex)
        fff = 0
        for i in range(Ncl):
            for j in range(Nray):
                H_quan += alpha_h_quan[b][fff] * np.dot(ar_quan[b][fff], np.conjugate(at_quan[b][fff]).T)
                fff += 1
        H_quan = np.sqrt(Nt * Nr / Ncl * Nray) * H_quan
        H_quan = c2m(H_quan)
        H_quan = H_quan.to(dtype=torch.float)
        HH_quan[b] = H_quan

    return HH,HH_quan

def H_gen_cgaussian(Batch_size):
    HH = torch.zeros((Batch_size, 2, Nr, Nt))
    # random seed
    for b in range(Batch_size):
        H = 1/math.sqrt(2)* (np.random.randn(Nr, Nt)+1j*np.random.randn(Nr, Nt))
        H = c2m(H)
        H = H.to(dtype=torch.float)
        HH[b] = H
    
    HH_quan = torch.zeros((Batch_size, 2, Nr, Nt))
    HH_quan[:] = HH[:]
    length = Batch_size*Nr*Nt*2
    HH_quan = HH_quan.view(length)
    for i in range(length):
       
        HH_quan[i] = LloydQuan(bound_table_G, mapping_table_G, HH_quan[i], N_level=N_level)
   
    HH_quan = HH_quan.view(Batch_size, 2, Nr, Nt)
    return HH, HH_quan


def cal_nmse(sparse_gt, sparse_pred):
    # Calculate the NMSE
    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    return nmse

if __name__ == '__main__':
    Batch_Size = 10
    Num_Batch = 10
    Num_Pairs = Num_Batch*Batch_Size

    NMSE = 0
    H_Pairs_raw, H_Pairs_quan = H_gen_cgaussian(Batch_Size)
    nmse_batch = cal_nmse(H_Pairs_quan, H_Pairs_raw)
    NMSE += nmse_batch
    print('Currently finish producing 1 / %d, the nmse is %.3f dB. Time initialize.' %(Num_Batch, nmse_batch))

    start_time = time.time()
    for i in range(Num_Batch-1):
        H_raw, H_quan = H_gen_cgaussian(Batch_Size)
        H_Pairs_raw = torch.cat((H_Pairs_raw, H_raw))
        H_Pairs_quan = torch.cat((H_Pairs_quan, H_quan))
        nmse_batch = cal_nmse(H_raw, H_quan)
        NMSE += nmse_batch
        current_time = time.time() - start_time
        current_time_str = str(datetime.timedelta(seconds=int(current_time)))
        print('Currently finish producing %d / %d, the nmse is %.3f dB. Time: %s' %(i+2, Num_Batch, nmse_batch, current_time_str))
    NMSE /= Num_Batch
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    H_pairs = torch.cat((H_Pairs_raw.unsqueeze(0), H_Pairs_quan.unsqueeze(0)))
    H_pairs = H_pairs.numpy()
    print('Total time {}. Generate total {} samples.'.format(total_time_str, Num_Pairs))

    # np.save(f'data/Hpairs_mmWave/Hpairs_mmWave_NtNr_{Nt}_{Nr}_{quan_bits:2d}Quan_bits_{Num_Pairs}_samples.npy', H_pairs)
    np.save(f'data/Hpairs_Gaussian/Hpairs_Gaussian_NtNr_{Nt}_{Nr}_{quan_bits}Quan_bits_{Num_Pairs}_samples.npy', H_pairs)
    # print(cal_nmse(H_Pairs_quan, H_Pairs_raw))
    
    