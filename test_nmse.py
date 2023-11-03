import numpy as np
import torch
def cal_nmse(sparse_gt, sparse_pred):
    # Calculate the NMSE
    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    return nmse
def F_norm(H):
    H = H[:,0]+1j*H[:,1]
    return np.sqrt((np.sum(abs(H**2))))

#H_Pairs = np.load('data/Hpairs_Gaussian/Hpairs_Gaussian_NtNr_16_16_1Quan_bits_100_samples.npy')

H_Pairs = np.load('data/Hpairs_mmWave/Hpairs_mmWave_NtNr_32_32_7Quan_bits_200_samples.npy')
print(H_Pairs)
H_Pairs_raw, H_Pairs_quan = H_Pairs[0], H_Pairs[1]
H_Pairs_raw[53] = H_Pairs_raw[53]/F_norm(H_Pairs_raw[53])
print(F_norm(H_Pairs_raw[53]))
# print(H_Pairs.shape)
# nmse_batch = cal_nmse(torch.tensor(H_Pairs_raw), torch.tensor(H_Pairs_quan))  
# print(nmse_batch)

# H_Pairs = np.load('data/Hpairs_mmWave/Hpairs_mmWave_NtNr_16_16_6Quan_bits_200_samples.npy')
# H_Pairs_raw, H_Pairs_quan = H_Pairs[0], H_Pairs[1]
# nmse_batch = cal_nmse(torch.tensor(H_Pairs_raw), torch.tensor(H_Pairs_quan))  
# print(nmse_batch)





for i in [2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
    print(i)
    H_Pairs = np.load(f'data/Hpairs_mmWave/Hpairs_mmWave_NtNr_16_16_Cr_{i}_20000_samples.npy')
    H_Pairs_raw, H_Pairs_quan = H_Pairs[0], H_Pairs[1]
    #print(F_norm(H_Pairs_raw[30]))
    nmse_batch = cal_nmse(torch.tensor(H_Pairs_raw), torch.tensor(H_Pairs_quan))  
    #print(nmse_batch)
    
