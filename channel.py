import numpy as np
import math
import torch

class Channels():
    
    def AWGN(self, Tx_sig, n_std):
        device = Tx_sig.device
        noise = torch.normal(0, n_std/math.sqrt(2), size=Tx_sig.shape).to(device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_std):
        device = Tx_sig.device
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_std, K=1):
        device = Tx_sig.device
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig



Es_16qam = 10
mapping_table_16qam = {
    (1,0,1,0) : -3-3j,
    (1,0,1,1) : -3-1j,
    (1,0,0,1) : -3+1j,
    (1,0,0,0) : -3+3j,
    (1,1,1,0) : -1-3j,
    (1,1,1,1) : -1-1j,
    (1,1,0,1) : -1+1j,
    (1,1,0,0) : -1+3j,
    (0,1,1,0) : 1-3j,
    (0,1,1,1) : 1-1j,
    (0,1,0,1) : 1+1j,
    (0,1,0,0) : 1+3j,
    (0,0,1,0) : 3-3j,
    (0,0,1,1) : 3-1j,
    (0,0,0,1) : 3+1j,
    (0,0,0,0) : 3+3j,
}

Es_qpsk = 2
mapping_table_qpsk = {
    (0,0) :  1+1j,
    (0,1) :  1-1j,
    (1,0) : -1+1j,
    (1,1) : -1-1j,
}

bits_per_digit = 9

demapping_table_16qam = {v : k for k, v in mapping_table_16qam.items()}
demapping_table_qpsk = {v : k for k, v in mapping_table_qpsk.items()}


def SVD_Precoding(H, P, d):
    U,D,V = np.linalg.svd(H, full_matrices=True)
   
    W_svd = V.conj().transpose(0,2,1)[:,:,:d]
    M_svd = U

    # W_svd_norm = np.sqrt(np.trace(np.matmul(W_svd,W_svd.conj().transpose(0,2,1)), axis1=1, axis2=2))   #power norm
    # W_svd = W_svd * np.sqrt(P) / W_svd_norm
    return W_svd, D, M_svd
    
def SignalNorm(signal, P, mod_type=None):
    
    signal_power = np.mean(abs(signal**2), axis=(1,2), keepdims=True)

    if mod_type is not None:
        if mod_type=='QPSK':
            return signal / np.sqrt(Es_qpsk) 
        elif mod_type=='16QAM':
            return signal / np.sqrt(Es_16qam)
    else:
        return signal * np.sqrt(P) / np.sqrt(signal_power)

def SignalDenorm(signal, P, mod_type=None):
    signal_power = np.mean(abs(signal**2))
    if mod_type is not None:
        if mod_type=='QPSK':
            return signal * np.sqrt(Es_qpsk)
        elif mod_type=='16QAM':
            return signal * np.sqrt(Es_16qam)
    else:
        return signal * math.sqrt(P) / math.sqrt(signal_power)

def cal_mse(A, B):
    if A.shape==B.shape:
        return np.sum((A-B)**2)/len(A.flatten())
    elif (len(A.flatten())==len(B.flatten())):
        return np.sum((A-B.reshape(A.shape))**2)/len(A.flatten())
    else:
        raise ValueError("The shape is different")

def cal_mse_tensor(A, B):
    if A.shape==B.shape:
        return torch.sum((A-B)**2)/len(A.flatten())
    elif (len(A.flatten())==len(B.flatten())):
        return torch.sum((A-B.reshape(A.shape))**2)/len(A.flatten())
    else:
        raise ValueError("The shape is different")

def cal_nmse(sparse_gt, sparse_pred):
    # Calculate the NMSE
    power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
    difference = sparse_gt - sparse_pred
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    return nmse


class MIMO_Channel():
    def __init__(self, Nr=2, Nt=4, d=2, K=1, P=1):
        # Base configs
        self.Nt = Nt   # transmit antenna
        self.K = K     # users
        self.Nr = Nr   # receive antenna
        self.d = d     # data streams  ** d <= min(Nt/K, Nr)  **
        self.P = P     # power

        # mmWave configs
        # Nt = 32         # T antennas
        # Nr = 16         # R antennas
        self.NtRF = 4        # RF chains at the transmitter
        self.NrRF = 4        # RF chains at the receiver
        self.Ncl = 4         # clusters
        self.Nray = 6        # ray
        self.sigma_h = 0.3   # gain
        self.Tao = 0.001     # delay
        self.fd = 3          # maximum Doppler shift

    def Trans_Procedure_group(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        tx_times = int(total_num/self.d/2)
        symbol_group = Tx_sig.flatten().reshape(self.d, tx_times, 2)
        symbol_y = np.zeros_like(symbol_group)
        symbol_trans = symbol_group[:,:,0] + 1j*symbol_group[:,:,1]
        symbol_trans = SignalNorm(symbol_trans, self.P)
        y_de_group = np.zeros((self.d, tx_times), dtype=complex)
        symbol_x = symbol_trans

        noise = np.sqrt(sigma2/2) * (np.random.randn(self.K*self.Nr, tx_times)+1j*np.random.randn(self.K*self.Nr, tx_times)) 
        y= H.dot(V).dot(symbol_x) + noise    # y = HVx+n
        y_de = np.diag(1/D).dot(U.conj().T).dot(y) / self.P
        y_de = y_de[:self.d]
        y_de_group = y_de
        symbol_y[:,:,0] = np.real(SignalNorm(y_de_group, self.P))
        symbol_y[:,:,1] = np.imag(SignalNorm(y_de_group, self.P))
        return symbol_y.reshape(shape)
    
    def Trans_Procedure_element(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        tx_times = int(total_num/self.d/2)
        symbol_group = Tx_sig.flatten().reshape(tx_times, self.d, 1, 2)
        symbol_y = np.zeros_like(symbol_group)
        symbol_trans = symbol_group[:,:,:,0] + 1j*symbol_group[:,:,:,1]
        symbol_trans = SignalNorm(symbol_trans, self.P)
        y_de_group = np.zeros((tx_times, self.d, 1), dtype=complex)
        for i in range(tx_times):
            symbol_x = symbol_trans[i,:,:]
            noise = np.sqrt(sigma2/2) * (np.random.randn(self.K*self.Nr, 1)+1j*np.random.randn(self.K*self.Nr, 1))
            y= H.dot(V).dot(symbol_x) 
            y_de = np.diag(1/D).dot(U.conj().T).dot(y) / self.P
            y_de = y_de[:self.d]
            y_de_group[i] = y_de
        
        symbol_y[:,:,:,0] = np.real(SignalNorm(y_de_group, self.P))
        symbol_y[:,:,:,1] = np.imag(SignalNorm(y_de_group, self.P))
        return symbol_y.reshape(shape)

    def Circular_Gaussian(self, Tx_sig, snr=10, Group_enable=True):
        H = 1/math.sqrt(2)* (np.random.randn(self.Nr, self.Nt)+1j*np.random.randn(self.Nr, self.Nt))    # Nr * Nt
        V,_,_ = SVD_Precoding(H, self.P, self.d)
        U,D,_ = np.linalg.svd(H, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H, V, D, U, snr)
    
    def mmwave_MIMO(self, Tx_sig, snr=10, Group_enable=True):
        def theta(N, Seed=100):
            phi = np.zeros(self.Ncl*self.Nray)         # azimuth AoA and AoD
            a = np.zeros((self.Ncl*self.Nray, N, 1), dtype=complex)

            for i in range(self.Ncl*self.Nray):
                phi[i] = np.random.uniform(-np.pi/3, np.pi/3)
            f = 0
            for j in range(self.Ncl*self.Nray):
                f += 1
                for z in range(N):
                    a[j][z] = np.exp(1j * np.pi * z * np.sin(phi[f-1]))
            PHI = phi.reshape(self.Ncl*self.Nray)
            return a/np.sqrt(N), PHI
        
        def H_gen(Seed=100):
            HH = np.zeros((self.Nr, self.Nt))
            # complex gain
            alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray)) + 1j*np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray))
            # receive and transmit array response vectors
            ar, ThetaR = theta(self.Nr, Seed+10000)
            at, ThetaT = theta(self.Nt, Seed)
            H = np.zeros((self.Nr, self.Nt), dtype=complex)
            fff = 0
            for i in range(self.Ncl):
                for j in range(self.Nray):
                    H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)
                    # H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)*np.exp(1j*2*np.pi*Tao*fd*np.cos(ThetaR[fff]))    # channel with delay
                    fff += 1
            H = np.sqrt(self.Nt * self.Nr / self.Ncl * self.Nray) * H
            return H
        H = H_gen()   # Nr * Nt
        V,_,_ = SVD_Precoding(H, self.P, self.d)
        U,D,_ = np.linalg.svd(H, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H, V, D, U, snr)


def SignalNorm_tensor(signal, P, mod_type=None):
    signal_power = torch.mean(abs(signal**2))
    if mod_type is not None:
        if mod_type=='QPSK':
            return signal / torch.sqrt(Es_qpsk) 
        elif mod_type=='16QAM':
            return signal / torch.sqrt(Es_16qam)
    else:
        return signal * torch.sqrt(P) / torch.sqrt(signal_power)

def SignalDenorm_tensor(signal, P, mod_type=None):
    signal_power = torch.mean(abs(signal**2))
    if mod_type is not None:
        if mod_type=='QPSK':
            return signal * torch.sqrt(Es_qpsk)
        elif mod_type=='16QAM':
            return signal * torch.sqrt(Es_16qam)
    else:
        return signal * math.sqrt(P) / math.sqrt(signal_power)

def SVD_Precoding_tensor(H, P, d):
    U,D,V = torch.linalg.svd(H, full_matrices=True)
    W_svd = V.conj().T[:,:d]
    M_svd = U

    W_svd_norm = torch.sqrt(torch.trace(torch.mm(W_svd,W_svd.conj().T)))   #power norm
    W_svd = W_svd * np.sqrt(P) / W_svd_norm
    return W_svd, D, M_svd


def F_norm(H):
    H = H[:,0]+1j*H[:,1]
    return np.sqrt((np.sum(abs(H**2))))

class MIMO_Channel_Quan():
    def __init__(self, Nr=2, Nt=4, d=2, K=1, P=1):
        # Base configs
        self.Nt = Nt   # transmit antenna
        self.K = K     # users
        self.Nr = Nr   # receive antenna
        self.d = d     # data streams  ** d <= min(Nt/K, Nr)  **
        self.P = P     # power

        # mmWave configs
        # Nt = 32         # T antennas
        # Nr = 16         # R antennas
        self.NtRF = 4        # RF chains at the transmitter
        self.NrRF = 4        # RF chains at the receiver
        self.Ncl = 2         # clusters
        self.Nray = 2        # ray
        self.sigma_h = 0.3   # gain
        self.Tao = 0.001     # delay
        self.fd = 3          # maximum Doppler shift

    def Trans_Procedure_group(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        P_ele = self.P / self.Nt
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        imgs_num = Tx_sig.shape[0]
        tx_times = int(total_num//self.d//2//imgs_num)  
        
        symbol_group = Tx_sig.flatten().reshape(imgs_num, self.d, tx_times, 2)
        symbol_y = np.zeros_like(symbol_group)
        symbol_trans = symbol_group[:,:,:,0] + 1j*symbol_group[:,:,:,1]
        y_de_group = np.zeros((imgs_num, self.d, tx_times), dtype=complex)
        symbol_x = symbol_trans

        noise = np.sqrt(sigma2/2) * (np.random.randn(imgs_num, self.K*self.Nr, tx_times)+1j*np.random.randn(imgs_num, self.K*self.Nr, tx_times)) 

        
        V_x = SignalNorm(np.matmul(V,symbol_x), P_ele)
        y= np.matmul(H, V_x) + noise
        
        Digmatrix = np.zeros_like(H)
        for i in range(imgs_num):
            Digmatrix[i] = np.diag(1/D[i])
        
        y_de = np.matmul(np.matmul(Digmatrix, U.conj().transpose(0,2,1)),y)
        
        y_de = y_de[:,:self.d,:]
        y_de_group = y_de

        symbol_y[:,:,:,0] = np.real(SignalNorm(y_de_group, 1))
        symbol_y[:,:,:,1] = np.imag(SignalNorm(y_de_group, 1))
        return symbol_y.reshape(shape)
    
    def Trans_Procedure_element(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        P_ele = self.P / self.Nt
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        tx_times = int(total_num/self.d/2)
        symbol_group = Tx_sig.flatten().reshape(tx_times, self.d, 1, 2)
        symbol_y = np.zeros_like(symbol_group)
        symbol_trans = symbol_group[:,:,:,0] + 1j*symbol_group[:,:,:,1]
        y_de_group = np.zeros((tx_times, self.d, 1), dtype=complex)
        for i in range(tx_times):
            symbol_x = symbol_trans[i,:,:]
            noise = np.sqrt(sigma2/2) * (np.random.randn(self.K*self.Nr, 1)+1j*np.random.randn(self.K*self.Nr, 1))
            V_x = SignalNorm(V.dot(symbol_x), P_ele)
            y= H.dot(V_x) + noise
            y_de = np.diag(1/D).dot(U.conj().T).dot(y)
            y_de = y_de[:self.d]
            y_de_group[i] = y_de
        symbol_y[:,:,:,0] = np.real(SignalNorm(y_de_group, 1))
        symbol_y[:,:,:,1] = np.imag(SignalNorm(y_de_group, 1))
        return symbol_y.reshape(shape)



    def Circular_Gaussian(self, Tx_sig, snr=10, Group_enable=False):
        H_raw = 1/math.sqrt(2)* (np.random.randn(self.Nr, self.Nt)+1j*np.random.randn(self.Nr, self.Nt))    # Nr * Nt
        sigmaH = self.P * 10**(-1000/10) 
        H_raw = H_raw/F_norm(H_raw)
        H_quan = H_raw 
        V,_,_ = SVD_Precoding(H_quan, self.P, self.d)
        U,D,_ = np.linalg.svd(H_raw, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)
    
    def mmwave_MIMO(self, Tx_sig, snr=10, Group_enable=True):
        def theta(N, Seed=100):
            phi = np.zeros(self.Ncl*self.Nray)         # azimuth AoA and AoD
            a = np.zeros((self.Ncl*self.Nray, N, 1), dtype=complex)

            for i in range(self.Ncl*self.Nray):
                phi[i] = np.random.uniform(-np.pi/3, np.pi/3)
            f = 0
            for j in range(self.Ncl*self.Nray):
                f += 1
                for z in range(N):
                    a[j][z] = np.exp(1j * np.pi * z * np.sin(phi[f-1]))
            PHI = phi.reshape(self.Ncl*self.Nray)
            return a/np.sqrt(N), PHI
        
        def H_gen(Seed=100):
            HH = np.zeros((self.Nr, self.Nt))
            # complex gain
            alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray)) + 1j*np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray))
            # receive and transmit array response vectors
            ar, ThetaR = theta(self.Nr, Seed+10000)
            at, ThetaT = theta(self.Nt, Seed)
            H = np.zeros((self.Nr, self.Nt), dtype=complex)
            fff = 0
            for i in range(self.Ncl):
                for j in range(self.Nray):
                    H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)
                    # H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)*np.exp(1j*2*np.pi*Tao*fd*np.cos(ThetaR[fff]))    # channel with delay
                    fff += 1
            H = np.sqrt(self.Nt * self.Nr / self.Ncl * self.Nray) * H
            return H
        H_raw = H_gen()   # Nr * Nt
        sigmaH = self.P * 10**(-10000/10) 
        
        # H_raw = np.ones((self.Nr,self.Nt))
        H_raw = H_raw/F_norm(H_raw)
        H_quan = H_raw + np.random.normal(0, sigmaH, (self.Nr, self.Nt)) + 1j*np.random.normal(0, sigmaH, (self.Nr, self.Nt))
        V,_,_ = SVD_Precoding(H_quan, self.P, self.d)
        U,D,_ = np.linalg.svd(H_raw, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)

    def Sim_Quan(self, Tx_sig, H_raw=None, H_quan=None, snr=20, Group_enable=True):
        H_raw = H_raw[:,0] + 1j*H_raw[:,1]
        H_quan = H_quan[:,0] + 1j*H_quan[:,1]
        sigmaH = self.P * 10**(-10/10) 
        # np.random.seed(10)
        # H_quan = H_raw + np.random.normal(0, sigmaH, (5, self.Nr, self.Nt)) + 1j*np.random.normal(0, sigmaH, (5, self.Nr, self.Nt))
        # np.random.seed(110)
        V,_,_ = SVD_Precoding(H_quan, self.P, self.d)
        U,D,_ = np.linalg.svd(H_quan, full_matrices=False)
  
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)


class MIMO_Channel_Tensor():
    def __init__(self, Nr=2, Nt=4, d=2, K=1, P=1, device=None):
        # Base configs
        self.Nt = Nt   # transmit antenna
        self.K = K     # users
        self.Nr = Nr   # receive antenna
        self.d = d     # data streams  ** d <= min(Nt/K, Nr)  **
        self.P = P     # power

        self.device = device
        # mmWave configs
        # Nt = 32         # T antennas
        # Nr = 16         # R antennas
        self.NtRF = 4        # RF chains at the transmitter
        self.NrRF = 4        # RF chains at the receiver
        self.Ncl = 4         # clusters
        self.Nray = 6        # ray
        self.sigma_h = 0.3   # gain
        self.Tao = 0.001     # delay
        self.fd = 3          # maximum Doppler shift

    def Trans_Procedure_group(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        tx_times = int(total_num/self.d/2)
        symbol_group = Tx_sig.flatten().reshape(self.d, tx_times, 2).to(self.device)
        symbol_y = torch.zeros_like(symbol_group).to(self.device)
        symbol_trans = symbol_group[:,:,0] + 1j*symbol_group[:,:,1]

        symbol_trans = SignalNorm_tensor(symbol_trans, self.P)
    
        noise = math.sqrt(sigma2/2) * (torch.randn(self.K*self.Nr, tx_times)+1j*torch.randn(self.K*self.Nr, tx_times)).to(self.device)

        y = torch.mm(torch.mm(H,V),symbol_trans)
        y_de = torch.mm(torch.mm(torch.diag(1/D)+1j*0,(U.conj().T)),y) + noise
        y_de_group = y_de[:self.d]
        symbol_y[:,:,0] = torch.real(SignalNorm_tensor(y_de_group, 1))
        symbol_y[:,:,1] = torch.imag(SignalNorm_tensor(y_de_group, 1))
        return symbol_y.reshape(shape)

    def Trans_Procedure_element(self, Tx_sig, H, V, D, U, snr=20):
        sigma2 = self.P * 10**(-snr/10) 
        total_num = len(Tx_sig.flatten())
        shape = Tx_sig.shape
        tx_times = int(total_num/self.d/2)
        symbol_group = Tx_sig.flatten().reshape(tx_times, self.d, 1, 2).to(self.device)
        symbol_y = torch.zeros_like(symbol_group).to(self.device)
        symbol_trans = symbol_group[:,:,:,0] + 1j*symbol_group[:,:,:,1]
        symbol_trans = SignalNorm_tensor(symbol_trans, self.P).to(self.device)
        y_de_group = (torch.zeros((tx_times, self.d, 1))+1j*0).to(self.device)
    
        for i in range(tx_times):
            symbol_x = symbol_trans[i,:,:]
            noise = math.sqrt(sigma2/2) * (torch.randn(self.K*self.Nr, 1)+1j*torch.randn(self.K*self.Nr, 1)).to(self.device)
            y= torch.mm(torch.mm(H,V),symbol_x) + noise
            y_de = torch.mm(torch.mm(torch.diag(1/D)+1j*0,(U.conj().T)),y) 
            y_de = y_de[:self.d]
            y_de_group[i] = y_de
        symbol_y[:,:,:,0] = torch.real(SignalNorm_tensor(y_de_group, 1))
        symbol_y[:,:,:,1] = torch.imag(SignalNorm_tensor(y_de_group, 1))
        return symbol_y.reshape(shape)

    def Circular_Gaussian(self, Tx_sig, snr=10, Group_enable=True):
        H_raw = 1/math.sqrt(2)* (torch.randn((self.Nr, self.Nt))+1j*torch.randn(self.Nr, self.Nt))    # Nr * Nt
        H_raw = torch.tensor(H_raw).to(self.device)
        sigmaH = self.P * 10**(-snr/10) 
        H_quan = H_raw + math.sqrt(sigmaH/2)* (torch.randn(self.Nr, self.Nt)+1j*torch.randn(self.Nr, self.Nt)).to(self.device)
        # H_quan.to(self.device)
        V,_,_ = SVD_Precoding_tensor(H_quan, self.P, self.d)
        U,D,_ = torch.linalg.svd(H_raw, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)
    
    def mmwave_MIMO(self, Tx_sig, snr=10, Group_enable=True):
        def theta(N, Seed=100):
            phi = np.zeros(self.Ncl*self.Nray)         # azimuth AoA and AoD
            a = np.zeros((self.Ncl*self.Nray, N, 1), dtype=complex)

            for i in range(self.Ncl*self.Nray):
                phi[i] = np.random.uniform(-np.pi/3, np.pi/3)
            f = 0
            for j in range(self.Ncl*self.Nray):
                f += 1
                for z in range(N):
                    a[j][z] = np.exp(1j * np.pi * z * np.sin(phi[f-1]))
            PHI = phi.reshape(self.Ncl*self.Nray)
            return a/np.sqrt(N), PHI
        
        def H_gen(Seed=100):
            HH = np.zeros((self.Nr, self.Nt))
            # complex gain
            alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray)) + 1j*np.random.normal(0, self.sigma_h, (self.Ncl*self.Nray))
            # receive and transmit array response vectors
            ar, ThetaR = theta(self.Nr, Seed+10000)
            at, ThetaT = theta(self.Nt, Seed)
            H = np.zeros((self.Nr, self.Nt), dtype=complex)
            fff = 0
            for i in range(self.Ncl):
                for j in range(self.Nray):
                    H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)
                    # H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)*np.exp(1j*2*np.pi*Tao*fd*np.cos(ThetaR[fff]))    # channel with delay
                    fff += 1
            H = np.sqrt(self.Nt * self.Nr / self.Ncl * self.Nray) * H
            # H = c2m(H)
            return torch.tensor(H)
        H_raw = H_gen()   # Nr * Nt
        sigmaH = self.P * 10**(-snr/10) 
        H_quan = H_raw + math.sqrt(sigmaH/2)* (torch.randn(self.Nr, self.Nt)+1j*torch.randn(self.Nr, self.Nt))
        V,_,_ = SVD_Precoding_tensor(H_quan, self.P, self.d)
        U,D,_ = torch.linalg.svd(H_raw, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)

    def Sim_Quan(self, Tx_sig, H_raw=None, H_quan=None, snr=20, Group_enable=True):
    
        H_raw = H_raw[0] + 1j*H_raw[1]
        H_raw = torch.tensor(H_raw).to(self.device)
        H_quan = H_quan[0] + 1j*H_quan[1]
        H_quan = torch.tensor(H_quan).to(self.device)

        V,_,_ = SVD_Precoding_tensor(H_quan, self.P, self.d)
        U,D,_ = torch.linalg.svd(H_raw, full_matrices=False)
        if Group_enable:
            return self.Trans_Procedure_group(Tx_sig, H_raw, V, D, U, snr)
        else:
            return self.Trans_Procedure_element(Tx_sig, H_raw, V, D, U, snr)