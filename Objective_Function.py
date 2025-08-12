import numpy as np
from scipy.signal import savgol_filter
from Global_vars import Global_vars
from PKC import PrimaryKeyEncryptor
from tinyec import registry

def Objective_Function(soln):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))
    Data = Global_vars.Data
    soln = np.array(soln)
    for k in range(v):
        if soln.ndim == 2:
            sol = np.round(soln[k, :]).astype(np.int16)
        else:
            sol = np.round(soln).astype(np.int16)
        datas = np.zeros((Data.shape[0],Data.shape[1]))
        for a in range(Data.shape[0]):
            dat = []
            for b in range(Data.shape[1]):
                privkey = sol[:Global_vars.KeyLength].astype(np.str)
                privkey = int(''.join(privkey))
                Global_vars.Curve = registry.get_curve('brainpoolP160r1')  # ('brainpoolP256r1')
                Data = Global_vars.Data
                PrimaryKeyEncryptor(str(privkey)[:32]).encrypt(int(Data[a, b]))
                enc = PrimaryKeyEncryptor('134810bef265eabbfcc33c8956b9c870').encrypt(int(Data[a, b]))
                dec = PrimaryKeyEncryptor('134810bef265eabbfcc33c8956b9c870').decrypt(enc)
                dat.append(dec)
            EbN0dBs = -2
            M = 2  # Number of points in BPSK constellation
            m = np.arange(0, max(Data[:, 1]))  # all possible input symbols
            A = 1  # amplitude
            constellation = A * np.cos(m / M * 2 * np.pi)  # reference constellation for BPSK
            # ------------ Transmitter---------------
            inputSyms = dat  # Random 1's and 0's as input to BPSK modulator
            s = constellation[inputSyms]  # modulated symbols
            # ----------- Channel --------------
            # Compute power in modulatedSyms and add AWGN noise for given SNRs
            gamma = 10 ** (EbN0dBs / 10)  # SNRs to linear scale
            P = sum(abs(s) ** 2) / len(s)  # Actual power in the vector
            N0 = P / gamma  # Find the noise spectral density
            n = np.sqrt(N0 / 2) * np.random.standard_normal(s.shape)  # computed noise vector
            r = s + n  # received signal
            # -------------- Receiver ------------
            detectedSyms = (r <= 0).astype(int)  # thresolding at value 0
            noise_red = savgol_filter(detectedSyms, 3, 2)
            datas[a, :] = noise_red
        fitn[k] = np.var(datas)
    return fitn

