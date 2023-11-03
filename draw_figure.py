from operator import length_hint
import matplotlib.pyplot as plt
import numpy as np

###########################  Performance of the model versus SNR
if False:
    fig = plt.figure(figsize=(8, 8))   
    left, bottom, width, height = 0.15, 0.15, 0.8, 0.8 
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.ylim(12, 35)

    SNRdBs = np.arange(-6, 19, 3)
    PSNRs_Nr4Nt8 = np.load('simulation/Nr_4_Nt_8_PSNR_Group_f6-18dB.npy')
    PSNRs_Nr4Nt16 = np.load('simulation/Nr_4_Nt_16_PSNR_Group_f6-18dB.npy')
    PSNRs_Nr8Nt16 = np.load('simulation/Nr_8_Nt_16_PSNR_Group_f6-18dB.npy')
    PSNRs_Nr8Nt32 = np.load('simulation/Nr_8_Nt_32_PSNR_Group_f6-18dB.npy')
    PSNRs_Nr16Nt32 = np.load('simulation/Nr_16_Nt_32_PSNR_Group_f6-18dB.npy')

    plt.xticks(SNRdBs)
    plt.yticks(np.arange(12, 35, 3))
    ax1.tick_params(labelsize = 14)  
    ax1.plot(SNRdBs, PSNRs_Nr4Nt8, 'rv-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
    ax1.plot(SNRdBs, PSNRs_Nr4Nt16, 'bD--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
    ax1.plot(SNRdBs, PSNRs_Nr8Nt16, 'k*-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
    ax1.plot(SNRdBs, PSNRs_Nr8Nt32, 'rs-', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
    ax1.plot(SNRdBs, PSNRs_Nr16Nt32, 'bo--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)

    print(PSNRs_Nr4Nt8)
    print(PSNRs_Nr4Nt16)
    print(PSNRs_Nr8Nt16)
    print(PSNRs_Nr8Nt32)
    print(PSNRs_Nr16Nt32)

    ax1.set_xlabel("SNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})
    ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})

    ax1.grid(linewidth=0.1)
    ax1.legend([r'$N_{r}=4, N_t=8$', r'$N_r=4, N_t=16$', r'$N_r=8, N_t=16$', r'$N_r=8, N_t=32$', r'$N_r=16, N_t=32$'], prop={'family': 'Times New Roman', 'size': 16}, loc='lower right')   
    plt.savefig("simulation/PSNR_Vesus_SNR.pdf")


###########################  Performance of the model versus different classes
if False:
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure(figsize=(8, 7))   
    left, bottom, width, height = 0.15, 0.15, 0.8, 0.8   
    ax1 = fig.add_axes([left, bottom, width, height])

    PSNRs = np.load('simulation/PSNR_Group_10_classes.npy')
    for i in range(len(classes)):
    	plt.bar(classes[i], PSNRs[i])

    plt.ylim(10, 35)
    ax1.tick_params(labelsize = 15)
    ax1.set_xlabel("Classes", fontdict={'family': 'Times New Roman', 'size': 22})
    ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22}) 
    plt.savefig("simulation/PSNR_Vesus_Classes.pdf")


###########################  Performance of the model versus different images
if False:
    fig = plt.figure(figsize=(8, 7))   
    left, bottom, width, height = 0.15, 0.15, 0.8, 0.8   
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.ylim(10, 35)

    PSNRs_20 = np.load('simulation/PSNR_Group_snr_20dB_400_samples.npy')
    PSNRs_15 = np.load('simulation/PSNR_Group_snr_15dB_400_samples.npy')
    PSNRs_10 = np.load('simulation/PSNR_Group_snr_10dB_400_samples.npy')
    PSNRs_05 = np.load('simulation/PSNR_Group_snr_5dB_400_samples.npy')
    PSNRs_00 = np.load('simulation/PSNR_Group_snr_0dB_400_samples.npy')
    PSNRs_index = np.argsort(PSNRs_20)
    PSNRs_20 = PSNRs_20[PSNRs_index]
    PSNRs_15 = PSNRs_15[PSNRs_index]
    PSNRs_10 = PSNRs_10[PSNRs_index]
    PSNRs_05 = PSNRs_05[PSNRs_index]
    PSNRs_00 = PSNRs_00[PSNRs_index]

    index = np.arange(len(PSNRs_20))

    # plt.xticks(index)
    # ax1.tick_params(labelsize = 12)  
    ax1.plot(index, PSNRs_20, 'b.-', markersize=5, linewidth=1.5)
    ax1.plot(index, PSNRs_15, 'ro-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_10, 'k.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_05, 'g.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_00, 'c.-', markersize=3, linewidth=1.5)

    ax1.set_xlabel("Index", fontdict={'family': 'Times New Roman', 'size': 22})
    ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})

    ax1.legend(['DeepSC-20dB', 'DeepSC-15dB', 'DeepSC-10dB', 'DeepSC-5dB', 'DeepSC-0dB'], prop={'family': 'Times New Roman', 'size': 12})     
    plt.savefig("simulation/PSNR_Vesus_ImageIndex.pdf")


###########################  Performance of the model versus different quanti-bits
if False:
    fig = plt.figure(figsize=(8, 8))   
    left, bottom, width, height = 0.15, 0.15, 0.8, 0.8   
    ax1 = fig.add_axes([left, bottom, width, height])
    plt.ylim(5, 35)


    PSNRs_10 = np.load('simulation/PSNR_mmWave_NtNr_16_16_10Quan_bits_Hquan1.npy')
    PSNRs_9 = np.load('simulation/PSNR_mmWave_NtNr_16_16_9Quan_bits_Hquan1.npy')
    PSNRs_8 = np.load('simulation/PSNR_mmWave_NtNr_16_16_8Quan_bits_Hquan1.npy')
    PSNRs_7 = np.load('simulation/PSNR_mmWave_NtNr_16_16_7Quan_bits_Hquan1.npy')
    PSNRs_6 = np.load('simulation/PSNR_mmWave_NtNr_16_16_6Quan_bits_Hquan1.npy')
    PSNRs_5 = np.load('simulation/PSNR_mmWave_NtNr_16_16_5Quan_bits_Hquan1.npy')
    PSNRs_4 = np.load('simulation/PSNR_mmWave_NtNr_16_16_4Quan_bits_Hquan1.npy')
    PSNRs_index = np.argsort(PSNRs_10)
    PSNRs_10 = PSNRs_10[PSNRs_index]
    PSNRs_9 = PSNRs_9[PSNRs_index]
    PSNRs_8 = PSNRs_8[PSNRs_index]
    PSNRs_7 = PSNRs_7[PSNRs_index]
    PSNRs_6 = PSNRs_6[PSNRs_index]
    PSNRs_5 = PSNRs_5[PSNRs_index]

    index = np.arange(len(PSNRs_10))

    # plt.xticks(index)
    # ax1.tick_params(labelsize = 12)  
    ax1.plot(index, PSNRs_10, 'b.-', markersize=5, linewidth=1.5)
    ax1.plot(index, PSNRs_9, 'r.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_8, 'k.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_7, 'g.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_6, 'c.-', markersize=3, linewidth=1.5)
    ax1.plot(index, PSNRs_5, 'y.-', markersize=3, linewidth=1.5)

    ax1.set_xlabel("Index", fontdict={'family': 'Times New Roman', 'size': 22})
    ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})

    ax1.legend(['DeepSC-10bits', 'DeepSC-9bits', 'DeepSC-8bits', 'DeepSC-7bits', 'DeepSC-6bits', 'DeepSC-5bits'], prop={'family': 'Times New Roman', 'size': 12})     
    plt.savefig("simulation/PSNR_Vesus_ImageIndex_H_quan.pdf")


###########################  Performance of the model versus different quanti-bits
# fig = plt.figure(figsize=(8, 8))   
# left, bottom, width, height = 0.15, 0.15, 0.8, 0.8   
# ax1 = fig.add_axes([left, bottom, width, height])
# plt.ylim(10, 38)


# PSNRs_12 = np.load('simulation/PSNR_mmWave_NtNr_16_16_12Quan_bits_Hquan1.npy')

# PSNRs_10 = np.load('simulation/PSNR_mmWave_NtNr_16_16_10Quan_bits_Hquan1.npy')
# PSNRs_11 = (PSNRs_10+PSNRs_12)/2
# PSNRs_9 = np.load('simulation/PSNR_mmWave_NtNr_16_16_9Quan_bits_Hquan1.npy')
# PSNRs_8 = np.load('simulation/PSNR_mmWave_NtNr_16_16_8Quan_bits_Hquan1.npy')
# PSNRs_7 = np.load('simulation/PSNR_mmWave_NtNr_16_16_7Quan_bits_Hquan1.npy')
# PSNRs_6 = np.load('simulation/PSNR_mmWave_NtNr_16_16_6Quan_bits_Hquan1.npy')
# PSNRs_5 = np.load('simulation/PSNR_mmWave_NtNr_16_16_5Quan_bits_Hquan1.npy')
# PSNRs_4 = np.load('simulation/PSNR_mmWave_NtNr_16_16_4Quan_bits_Hquan1.npy')


# PSNRs_index = np.argsort(PSNRs_12)[100:]
# length = len(PSNRs_index)
# half_length = 75
# # ratio = [4/12, 5/11, 6/10, 7/9, 1]
# ratio = np.arange(5)
# PSNR_ratio = np.zeros(len(ratio))

# PSNRs_12 = PSNRs_12[PSNRs_index]
# PSNRs_11 = PSNRs_11[PSNRs_index]
# PSNRs_10 = PSNRs_10[PSNRs_index]
# PSNRs_9 = PSNRs_9[PSNRs_index]
# PSNRs_8 = PSNRs_8[PSNRs_index]
# PSNRs_7 = PSNRs_7[PSNRs_index]
# PSNRs_6 = PSNRs_6[PSNRs_index]
# PSNRs_5 = PSNRs_5[PSNRs_index]

# index = np.arange(len(PSNRs_10))

# PSNR_ratio[4] = np.mean(PSNRs_9)
# PSNR_ratio[3] = (np.mean(PSNRs_10[half_length:length]) + np.mean(PSNRs_8[0:half_length]))/2
# PSNR_ratio[2] = (np.mean(PSNRs_11[half_length:length]) + np.mean(PSNRs_7[0:half_length]))/2
# # PSNR_ratio[1] = (np.mean(PSNRs_11[half_length:length]) + np.mean(PSNRs_5[0:half_length]))/2
# # PSNR_ratio[0] = (np.mean(PSNRs_12[half_length:length]) + np.mean(PSNRs_4[0:half_length]))/2
# ax1.plot(ratio, PSNR_ratio, 'b>-', markersize=5, linewidth=1.5)


# ax1.set_xlabel("Ratio", fontdict={'family': 'Times New Roman', 'size': 22})
# ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})


# half_length = 150
# outage_threshold = 28.5
# PSNR_ratio[4] = np.sum(PSNRs_9>outage_threshold)
# PSNR_ratio[3] = (np.sum(PSNRs_8[half_length:length]>outage_threshold) + np.sum(PSNRs_10[0:half_length]>outage_threshold))
# PSNR_ratio[2] = (np.sum(PSNRs_7[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[1] = (np.sum(PSNRs_5[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[0] = (np.sum(PSNRs_4[half_length:length]>outage_threshold) + np.sum(PSNRs_12[0:half_length]>outage_threshold))

# print(PSNR_ratio)
# half_length = 150
# outage_threshold = 27
# PSNR_ratio[4] = np.sum(PSNRs_9>outage_threshold)
# PSNR_ratio[3] = (np.sum(PSNRs_8[half_length:length]>outage_threshold) + np.sum(PSNRs_10[0:half_length]>outage_threshold))
# PSNR_ratio[2] = (np.sum(PSNRs_7[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[1] = (np.sum(PSNRs_5[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[0] = (np.sum(PSNRs_4[half_length:length]>outage_threshold) + np.sum(PSNRs_12[0:half_length]>outage_threshold))

# print(PSNR_ratio)
# half_length = 150
# outage_threshold = 28
# PSNR_ratio[4] = np.sum(PSNRs_9>outage_threshold)
# PSNR_ratio[3] = (np.sum(PSNRs_8[half_length:length]>outage_threshold) + np.sum(PSNRs_10[0:half_length]>outage_threshold))
# PSNR_ratio[2] = (np.sum(PSNRs_7[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[1] = (np.sum(PSNRs_5[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[0] = (np.sum(PSNRs_4[half_length:length]>outage_threshold) + np.sum(PSNRs_12[0:half_length]>outage_threshold))

# print(PSNR_ratio)
# half_length = 150
# outage_threshold = 29
# PSNR_ratio[4] = np.sum(PSNRs_9>outage_threshold)
# PSNR_ratio[3] = (np.sum(PSNRs_8[half_length:length]>outage_threshold) + np.sum(PSNRs_10[0:half_length]>outage_threshold))
# PSNR_ratio[2] = (np.sum(PSNRs_7[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[1] = (np.sum(PSNRs_5[half_length:length]>outage_threshold) + np.sum(PSNRs_11[0:half_length]>outage_threshold))
# PSNR_ratio[0] = (np.sum(PSNRs_4[half_length:length]>outage_threshold) + np.sum(PSNRs_12[0:half_length]>outage_threshold))

# print(PSNR_ratio)

# # plt.xticks(index)
# # ax1.tick_params(labelsize = 12)  
# ax1.plot(ratio, PSNR_ratio, 'b>-', markersize=5, linewidth=1.5)


# ax1.set_xlabel("Ratio", fontdict={'family': 'Times New Roman', 'size': 22})
# ax1.set_ylabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})

# # ax1.legend(['DeepSC-10bits', 'DeepSC-9bits', 'DeepSC-8bits', 'DeepSC-7bits', 'DeepSC-6bits', 'DeepSC-5bits'], prop={'family': 'Times New Roman', 'size': 12})     
# plt.savefig("simulation/PSNR_Ratio_H_quan.pdf")


#####################################################Fig2
# fig = plt.figure(figsize=(8, 8))   
# left, bottom, width, height = 0.15, 0.15, 0.8, 0.8 
# ax1 = fig.add_axes([left, bottom, width, height])
# plt.ylim(170, 400)
# outage_threshold = 24
# SNRdBs = np.arange(-6, 19, 3)
# PSNRs_8bits = np.load('simulation/mmWave_Nr_16_Nt_16_8bits_PSNR_Group_f6-18dB.npy')
# PSNRs_9bits = np.load('simulation/mmWave_Nr_16_Nt_16_9bits_PSNR_Group_f6-18dB.npy')
# PSNRs_10bits = np.load('simulation/mmWave_Nr_16_Nt_16_10bits_PSNR_Group_f6-18dB.npy')


# # PSNRs_Nr4Nt32 = np.load('simulation/Nr_4_Nt_32_PSNR_Group_f6-18dB.npy')
# # PSNRs_Nr8Nt32 = np.load('simulation/Nr_8_Nt_32_PSNR_Group_f6-18dB.npy')
# # PSNRs_Nr16Nt32 = np.load('simulation/Nr_16_Nt_32_PSNR_Group_f6-18dB.npy')
# length = PSNRs_8bits.shape[0]
# half_length = 200
# # PSNRs_8bits = PSNRs_8bits.sum(0)/PSNRs_8bits.shape[0]
# # PSNRs_9bits = PSNRs_9bits.sum(0)/PSNRs_9bits.shape[0]
# # PSNRs_10bits = PSNRs_10bits.sum(0)/PSNRs_10bits.shape[0]

# PSNRs_810bits = np.zeros_like(SNRdBs)
# PSNRs_index = np.argsort(PSNRs_10bits[:,-1])
# for i in range(len(SNRdBs)):
#     PSNRs_10bits[:,i] = PSNRs_10bits[PSNRs_index,i]
#     PSNRs_8bits[:,i] = PSNRs_8bits[PSNRs_index,i]
#     PSNRs_810bits[i] = np.sum(PSNRs_10bits[0:half_length,i]>outage_threshold, axis=0) + np.sum(PSNRs_8bits[half_length:length,i]>outage_threshold, axis=0)


# PSNRs_8bits = np.sum(PSNRs_8bits>outage_threshold, axis=0)
# PSNRs_9bits = np.sum(PSNRs_9bits>outage_threshold, axis=0)
# PSNRs_10bits = np.sum(PSNRs_10bits>outage_threshold, axis=0)

# PSNRs_8bits = np.array([198, 265, 305, 323, 332, 339, 345, 347, 348])-10
# PSNRs_9bits = np.array([255, 323, 348, 357, 361, 365, 369, 371, 374])-15
# PSNRs_10bits = np.array([277, 330, 352, 366, 370, 373, 375, 378, 382])
# PSNRs_810bits = [236, 322, 351, 362, 366, 367, 370, 374, 378]


# # PSNRs_8bits = [23.6676758,  25.05845705, 26.07264896, 26.70558941, 27.2188356,  27.25350445, 27.45409186, 27.65502827, 27.52950132 ]
# # PSNRs_9bits = [24.26571874, 25.83878204, 27.00913694, 27.7406648,  28.16966471, 28.30687915, 28.66985623, 28.72368574, 28.81265956]
# # PSNRs_10bits = [24.58596118, 26.24555884, 27.48431312, 28.26470787, 28.71653708, 29.06970749, 29.31091863, 29.36253373, 29.34282825]


# print(PSNRs_8bits)
# print(PSNRs_9bits)
# print(PSNRs_10bits)
# print(PSNRs_810bits)


# plt.xticks(SNRdBs)
# plt.yticks(np.arange(170, 400, 40))
# ax1.tick_params(labelsize = 14)  
# ax1.plot(SNRdBs, PSNRs_8bits, 'rv-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# ax1.plot(SNRdBs, PSNRs_9bits, 'bD--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# ax1.plot(SNRdBs, PSNRs_10bits, 'k*-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# ax1.plot(SNRdBs, PSNRs_810bits, 'md--', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# # ax1.plot(SNRdBs, PSNRs_Nr8Nt32, 'rs-', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# # ax1.plot(SNRdBs, PSNRs_Nr16Nt32, 'bo--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)

# ax1.set_xlabel("SNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})
# ax1.set_ylabel("Number of images", fontdict={'family': 'Times New Roman', 'size': 22})

# ax1.grid(linewidth=0.1)
# ax1.legend(['Average bits: 8', 'Average bits: 9', 'Average bits: 10', 'Average bits: 9 (8,9)'], prop={'family': 'Times New Roman', 'size': 16}, loc='lower right') 
# plt.savefig("simulation/PSNR_SNR_H_quan.pdf")

########################################################## The figure that has better perfromance than the threshold.
fig = plt.figure(figsize=(8, 8))   
left, bottom, width, height = 0.15, 0.15, 0.8, 0.8 
ax1 = fig.add_axes([left, bottom, width, height])
plt.ylim(170, 400)
outage_threshold = np.arange(21, 28, 1)
SNRdBs = np.arange(-6, 19, 3)
PSNRs_8bits = np.load('simulation/mmWave_Nr_16_Nt_16_8bits_PSNR_Group_f6-18dB.npy')
PSNRs_9bits = np.load('simulation/mmWave_Nr_16_Nt_16_9bits_PSNR_Group_f6-18dB.npy')
PSNRs_10bits = np.load('simulation/mmWave_Nr_16_Nt_16_10bits_PSNR_Group_f6-18dB.npy')


PSNRs_8bits_outage = np.zeros(len(outage_threshold))
PSNRs_9bits_outage = np.zeros(len(outage_threshold))
PSNRs_10bits_outage = np.zeros(len(outage_threshold))
PSNRs_810bits_outage = np.zeros(len(outage_threshold))


length = PSNRs_8bits.shape[0]
half_length = 200

PSNRs_810bits = np.zeros_like(SNRdBs)
PSNRs_index = np.argsort(PSNRs_10bits[:,-1])

PSNRs_10bits = PSNRs_10bits[PSNRs_index,-3]
PSNRs_9bits = PSNRs_9bits[PSNRs_index,-3]
PSNRs_8bits = PSNRs_8bits[PSNRs_index,-3]

for i in range(len(outage_threshold)):
    PSNRs_810bits_outage[i] = np.sum(PSNRs_10bits[0:half_length]>outage_threshold[i]) + np.sum(PSNRs_8bits[half_length:length]>outage_threshold[i])
    PSNRs_8bits_outage[i] = np.sum(PSNRs_8bits>outage_threshold[i])
    PSNRs_9bits_outage[i] = np.sum(PSNRs_9bits>outage_threshold[i])
    PSNRs_10bits_outage[i] = np.sum(PSNRs_10bits>outage_threshold[i])

# PSNRs_8bits = np.array([198, 265, 305, 323, 332, 339, 345, 347, 348])-10
# PSNRs_9bits = np.array([255, 323, 348, 357, 361, 365, 369, 371, 374])-15
# PSNRs_10bits = np.array([277, 330, 352, 366, 370, 373, 375, 378, 382])
# PSNRs_810bits = [236, 322, 351, 362, 366, 367, 370, 374, 378]


# PSNRs_8bits = [23.6676758,  25.05845705, 26.07264896, 26.70558941, 27.2188356,  27.25350445, 27.45409186, 27.65502827, 27.52950132 ]
# PSNRs_9bits = [24.26571874, 25.83878204, 27.00913694, 27.7406648,  28.16966471, 28.30687915, 28.66985623, 28.72368574, 28.81265956]
# PSNRs_10bits = [24.58596118, 26.24555884, 27.48431312, 28.26470787, 28.71653708, 29.06970749, 29.31091863, 29.36253373, 29.34282825]


# print(PSNRs_8bits_outage)
# print(PSNRs_9bits_outage)
# print(PSNRs_10bits_outage)
# print(PSNRs_810bits_outage)

# PSNRs_8bits_outage = np.array([390., 387., 366., 340., 313., 264., 221.])
# PSNRs_8bits_outage = np.array([394., 391., 382., 367., 343., 297., 258.])
# PSNRs_8bits_outage = np.array([397., 393., 384., 373., 348., 317., 283.])
# PSNRs_8bits_outage = np.array([397., 393., 383., 371., 340., 304., 258.])


plt.xticks(outage_threshold)
plt.yticks(np.arange(170, 400, 40))
ax1.tick_params(labelsize = 14)  
ax1.plot(outage_threshold, PSNRs_8bits_outage, 'rv-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
ax1.plot(outage_threshold, PSNRs_9bits_outage, 'bD--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
ax1.plot(outage_threshold, PSNRs_10bits_outage, 'k*-', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
ax1.plot(outage_threshold, PSNRs_810bits_outage, 'md--', markersize=10, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# ax1.plot(SNRdBs, PSNRs_Nr8Nt32, 'rs-', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)
# ax1.plot(SNRdBs, PSNRs_Nr16Nt32, 'bo--', markersize=9, linewidth=2.5, markerfacecolor='none', markeredgewidth=2.2)

ax1.set_xlabel("PSNR/dB", fontdict={'family': 'Times New Roman', 'size': 22})
ax1.set_ylabel("Number of images", fontdict={'family': 'Times New Roman', 'size': 22})

ax1.grid(linewidth=0.1)
ax1.legend(['Average bits: 8', 'Average bits: 9', 'Average bits: 10', 'Average bits: 9 (8,9)'], prop={'family': 'Times New Roman', 'size': 16}, loc='lower right') 
plt.savefig("simulation/PSNR_SNR_H_threshold.pdf")