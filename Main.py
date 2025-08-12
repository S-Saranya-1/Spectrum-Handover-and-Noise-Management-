import secrets
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter
import random as rn
from BlockChain import Blockchain
from DHOA import DHOA
from FFA import FFA
from Global_vars import Global_vars
from MPA import MPA
from Objective_Function import Objective_Function
from PKC import PrimaryKeyEncryptor
from PROPOSED import PROPOSED
from WOA import WOA
from numpy import matlib
from tinyec import registry
from Plot_Results import plot_results

# Read Dataset
an = 0
if an == 1:
    Directory = './Dataset/'
    dir1 = os.listdir(Directory)
    fold3 = []
    for i in range(1):
        file1 = Directory + dir1[i] + '/'
        dir2 = os.listdir(file1)

        for k in range(len(dir2)):
            file2 = file1 + dir2[k]
            if '_train' in file2:
                read = pd.read_csv(file2, sep='\t')
                np.save('Data.npy', read)

# Create Users and Trust level Computation
an = 0
if an == 1:
    Threshold = 0.5
    Users = np.zeros((200)).astype('int')
    Trust = np.zeros((200))
    for i in range(len(Users)):
        Users[i] = i + 1
        Trust[i] = np.random.rand()
    for j in range(len(Trust)):
        if Trust[j] > Threshold:
            Trust[j] = 1
        else:
            Trust[j] = 0
    np.save('Users.npy', Users)
    np.save('Trust.npy', Trust.astype('int'))

# Optimization for Encryption
an = 0
if an == 1:
    BestSol = []
    fitness = []
    Data = np.load('Data.npy', allow_pickle=True)
    Global_vars.Data = Data
    Global_vars.Curve = registry.get_curve('brainpoolP256r1')
    privKey = secrets.randbelow(Global_vars.Curve.field.n)
    Global_vars.Plain_Text = Data
    Global_vars.KeyLength = len(str(privKey))
    Npop = 10
    Chlen = len(str(privKey))
    xmin = matlib.repmat(np.concatenate([np.zeros((1, len(str(privKey))))], axis=None), Npop, 1)
    xmax = matlib.repmat(np.concatenate([9 * np.ones((1, len(str(privKey))))], axis=None), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Function
    max_iter = 25

    print('DHOA....')
    [bestfit1, fitness1, bestsol1, Time1] = DHOA(initsol, fname, xmin, xmax, max_iter)

    print('WOA....')
    [bestfit2, fitness2, bestsol2, Time2] = WOA(initsol, fname, xmin, xmax, max_iter)

    print('MPA....')
    [bestfit3, fitness3, bestsol3, Time3] = MPA(initsol, fname, xmin, xmax, max_iter)

    print('FFA....')
    [bestfit4, fitness4, bestsol4, Time4] = FFA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
    np.save('BestSol.npy', sol)
    np.save('Fitnesses.npy', fitness)

# Optimized
an = 0
if an == 1:
    trust = np.load('Trust.npy', allow_pickle=True)
    Data = np.load('Data.npy', allow_pickle=True)
    soln = np.load('BestSol.npy', allow_pickle=True)[4, :]
    datas = np.zeros((Data.shape[0], Data.shape[1]))
    sol = np.round(soln).astype(np.int16)
    for a in range(Data.shape[0]):
        print(a)
        dat = []
        for b in range(Data.shape[1]):
            Curve = registry.get_curve('brainpoolP256r1')
            privKey = secrets.randbelow(Curve.field.n)
            Plain_Text = Data
            KeyLength = len(str(privKey))
            privkey = sol[:KeyLength].astype(np.str)
            privkey = int(''.join(privkey))
            PrimaryKeyEncryptor(str(privkey)[:32]).encrypt(int(Data[a, b]))
            enc = PrimaryKeyEncryptor('134810bef265eabbfcc33c8956b9c870').encrypt(int(Data[a, b]))
            dec = PrimaryKeyEncryptor('134810bef265eabbfcc33c8956b9c870').decrypt(enc)
            dat.append(dec)
        EbN0dBs = -2
        M = 2  # Number of points in BPSK constellation
        m = np.arange(0, max(Data[:,1]+1))  # all possible input symbols
        A = 1  # amplitude
        constellation = A * np.cos(m / M * 2 * np.pi)  # reference constellation for BPSK
        # ------------ Transmitter---------------
        inputSyms = dat
        s = constellation[inputSyms]  # modulated symbols
        # ----------- Channel --------------
        # Compute power in modulatedSyms and add AWGN noise for given SNRs
        gamma = 10 ** (EbN0dBs / 10)  # SNRs to linear scale
        P = sum(abs(s) ** 2) / len(s)  # Actual power in the vector
        N0 = P / gamma  # Find the noise spectral density
        n = np.sqrt(N0 / 2) * np.random.standard_normal(s.shape)  # computed noise vector
        r = s + n  # received signal
        noise_red = savgol_filter(r, 3, 2)
        datas[a,:]=noise_red
    np.save('Noise_Removed_Data.npy', datas)

# Blockchain
an = 0
if an == 1:
    Blockchain()


plot_results()
