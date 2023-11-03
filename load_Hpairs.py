import numpy as np
H_Pairs_Dict = {}
Num_channels = {}

# for i in np.arange(12,4,-1):
#     # if i != 11:
#     H_Pairs = np.load(f'data/Hpairs_mmWave/Hpairs_mmWave_NtNr_16_16_{i}Quan_bits_10000_samples.npy')
#     H_Pairs_Dict[i] = H_Pairs
#     Num_channels[i] = H_Pairs[0].shape[0]
#     print('Finish loading the %d-th file of the mmWave Channel pairs' %(i))
#     del H_Pairs

# for i in np.arange(12,0,-1):
#     H_Pairs = np.load(f'data/Hpairs_Gaussian/Hpairs_Gaussian_NtNr_16_16_{i}Quan_bits_100_samples.npy')
#     H_Pairs_Dict[i] = H_Pairs
#     Num_channels[i] = H_Pairs[0].shape[0]
#     print('Finish loading the %d-th file of the Gaussian Channel pairs' %(i))
#     del H_Pairs


for i in np.arange(20,1,-1):
    # if i != 11:
    H_Pairs = np.load(f'data/Hpairs_mmWave/Hpairs_mmWave_NtNr_16_16_Cr_{i}_20000_samples.npy')
    H_Pairs_Dict[i] = H_Pairs
    Num_channels[i] = H_Pairs[0].shape[0]
    print('Finish loading the %d-th file of the mmWave Channel pairs' %(i))
    del H_Pairs

print(Num_channels) 
print(H_Pairs_Dict.keys) 
