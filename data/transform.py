import numpy as np

def compute_mean_energy(R_min, R_max):
    return np.sqrt(R_min * R_max) # TBD

def readfile(filename):
    x_min, x_max, y, yStaLo, yStaUp, ySysLo, ySysUp = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6), unpack=True)
    return x_min, x_max, y, yStaLo, yStaUp, ySysLo, ySysUp
    
def transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp, Z = 1.):
    Z = float(Z)
    R_mean = compute_mean_energy(R_min, R_max)
    E_mean = R_mean * Z
    I_E = I_R / Z
    IStaLo = eStaLo / Z
    IStaUp = eStaUp / Z
    ISysLo = eSysLo / Z
    ISysUp = eSysUp / Z
    return [E_mean, I_E, IStaLo, IStaUp, ISysLo, ISysUp]

def transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, A = 1.):
    A = float(A)
    T_mean = compute_mean_energy(T_min, T_max)
    E_mean = T_mean * A
    I_E = I_T / A
    IStaLo = eStaLo / A
    IStaUp = eStaUp / A
    ISysLo = eSysLo / A
    ISysUp = eSysUp / A
    return [E_mean, I_E, IStaLo, IStaUp, ISysLo, ISysUp]

def compute_AMS02_CNO():
    R_min_C, R_max_C, I_C, eStaLo_C, eStaUp_C, eSysLo_C, eSysUp_C = readfile('lake/AMS-02_C_rigidity.txt')
    R_min_N, R_max_N, I_N, eStaLo_N, eStaUp_N, eSysLo_N, eSysUp_N = readfile('lake/AMS-02_N_rigidity.txt')
    R_min_O, R_max_O, I_O, eStaLo_O, eStaUp_O, eSysLo_O, eSysUp_O = readfile('lake/AMS-02_O_rigidity.txt')

    R_min = R_min_O[0:63]
    R_max = R_max_O[0:63]
    I_R = I_O[0:63] + I_C[1:64] # + I_N[0:63]
    eStaLo = np.sqrt(eStaLo_C[1:64]**2. + eStaLo_O[0:63]**2.)
    eStaUp = np.sqrt(eStaUp_C[1:64]**2. + eStaUp_O[0:63]**2.)
    eSysLo = np.sqrt(eSysLo_C[1:64]**2. + eSysLo_O[0:63]**2.)
    eSysUp = np.sqrt(eSysUp_C[1:64]**2. + eSysUp_O[0:63]**2.)

    for i in range(62):
        assert(R_min[i] == R_min_C[i + 1])
        assert(R_min[i] == R_min_N[i])
        assert(R_min[i] == R_min_O[i])
        assert(R_max[i] == R_max_C[i + 1])
        assert(R_max[i] == R_max_N[i])
        assert(R_max[i] == R_max_O[i])

    return R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp

def compute_CALET_CNO():
    T_min_C, T_max_C, I_C, eStaLo_C, eStaUp_C, eSysLo_C, eSysUp_C = readfile('lake/CALET_C_kEnergyPerNucleon.txt')
    T_min_O, T_max_O, I_O, eStaLo_O, eStaUp_O, eSysLo_O, eSysUp_O = readfile('lake/CALET_O_kEnergyPerNucleon.txt')

    T_min = T_min_O
    T_max = T_max_O
    I_T = I_O + I_C
    eStaLo = np.sqrt(eStaLo_C**2. + eStaLo_O**2.)
    eStaUp = np.sqrt(eStaUp_C**2. + eStaUp_O**2.)
    eSysLo = np.sqrt(eSysLo_C**2. + eSysLo_O**2.)
    eSysUp = np.sqrt(eSysUp_C**2. + eSysUp_O**2.)

    for i in range(22):
        assert(T_min[i] == T_min_C[i])
        assert(T_min[i] == T_min_O[i])
        assert(T_max[i] == T_max_C[i])
        assert(T_max[i] == T_max_O[i])

    return T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp

def compute_CREAM_CNO():
    T_min_C, T_max_C, I_C, eStaLo_C, eStaUp_C, eSysLo_C, eSysUp_C = readfile('lake/CREAM_C_kEnergyPerNucleon.txt')
    T_min_O, T_max_O, I_O, eStaLo_O, eStaUp_O, eSysLo_O, eSysUp_O = readfile('lake/CREAM_O_kEnergyPerNucleon.txt')
    
    T_min = T_min_O[2:9]
    T_max = T_max_O[2:9]
    I_T = I_O[2:9] + I_C[2:9]
    eStaLo = np.sqrt(eStaLo_C[2:9]**2. + eStaLo_O[2:9]**2.)
    eStaUp = np.sqrt(eStaUp_C[2:9]**2. + eStaUp_O[2:9]**2.)
    eSysLo = np.sqrt(eSysLo_C[2:9]**2. + eSysLo_O[2:9]**2.)
    eSysUp = np.sqrt(eSysUp_C[2:9]**2. + eSysUp_O[2:9]**2.)

    for i in range(7):
        assert(T_min[i] == T_min_C[i + 2])
        assert(T_min[i] == T_min_O[i + 2])
        assert(T_max[i] == T_max_C[i + 2])
        assert(T_max[i] == T_max_O[i + 2])

    return T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp

def compute_CREAM_NeMgSi():
    T_min_Ne, T_max_Ne, I_Ne, eStaLo_Ne, eStaUp_Ne, eSysLo_Ne, eSysUp_Ne = readfile('lake/CREAM_Ne_kEnergyPerNucleon.txt')
    T_min_Mg, T_max_Mg, I_Mg, eStaLo_Mg, eStaUp_Mg, eSysLo_Mg, eSysUp_Mg = readfile('lake/CREAM_Mg_kEnergyPerNucleon.txt')
    T_min_Si, T_max_Si, I_Si, eStaLo_Si, eStaUp_Si, eSysLo_Si, eSysUp_Si = readfile('lake/CREAM_Si_kEnergyPerNucleon.txt')

    T_min = T_min_Ne[0:7]
    T_max = T_max_Ne[0:7]
    I_T = I_Ne[0:7] + I_Mg[1:8] + I_Si[1:8]
    eStaLo = np.sqrt(eStaLo_Ne[0:7]**2. + eStaLo_Mg[1:8]**2. + eStaLo_Si[1:8]**2.)
    eStaUp = np.sqrt(eStaUp_Ne[0:7]**2. + eStaUp_Mg[1:8]**2. + eStaUp_Si[1:8]**2.)
    eSysLo = np.sqrt(eSysLo_Ne[0:7]**2. + eSysLo_Mg[1:8]**2. + eSysLo_Si[1:8]**2.)
    eSysUp = np.sqrt(eSysUp_Ne[0:7]**2. + eSysUp_Mg[1:8]**2. + eSysUp_Si[1:8]**2.)

    for i in range(7):
        assert(T_min[i] == T_min_Ne[i])
        assert(T_min[i] == T_min_Mg[i + 1])
        assert(T_min[i] == T_min_Si[i + 1])
        assert(T_max[i] == T_max_Ne[i])
        assert(T_max[i] == T_max_Mg[i + 1])
        assert(T_max[i] == T_max_Si[i + 1])

    return T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp
    
def compute_AMS02_NeMgSi():
    R_min_Ne, R_max_Ne, I_Ne, eStaLo_Ne, eStaUp_Ne, eSysLo_Ne, eSysUp_Ne = readfile('lake/AMS-02_Ne_rigidity.txt')
    R_min_Mg, R_max_Mg, I_Mg, eStaLo_Mg, eStaUp_Mg, eSysLo_Mg, eSysUp_Mg = readfile('lake/AMS-02_Mg_rigidity.txt')
    R_min_Si, R_max_Si, I_Si, eStaLo_Si, eStaUp_Si, eSysLo_Si, eSysUp_Si = readfile('lake/AMS-02_Si_rigidity.txt')

    R_min = R_min_Si
    R_max = R_max_Si
    I_R = I_Ne + I_Mg + I_Si
    eStaLo = np.sqrt(eStaLo_Ne**2. + eStaLo_Mg**2. + eStaLo_Si**2.)
    eStaUp = np.sqrt(eStaUp_Ne**2. + eStaUp_Mg**2. + eStaUp_Si**2.)
    eSysLo = np.sqrt(eSysLo_Ne**2. + eSysLo_Mg**2. + eSysLo_Si**2.)
    eSysUp = np.sqrt(eSysUp_Ne**2. + eSysUp_Mg**2. + eSysUp_Si**2.)

    for i in range(66):
        assert(R_min[i] == R_min_Ne[i])
        assert(R_min[i] == R_min_Mg[i])
        assert(R_min[i] == R_min_Si[i])
        assert(R_max[i] == R_max_Ne[i])
        assert(R_max[i] == R_max_Mg[i])
        assert(R_max[i] == R_max_Si[i])

    return R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp

def dump(data, filename):
    print(filename)
    f = open('output/' + filename, 'w')
    E_mean, I_E, IStaLo, IStaUp, ISysLo, ISysUp = data
    for i,j,k,l,m,n in zip(E_mean, I_E, IStaLo, IStaUp, ISysLo, ISysUp):
        f.write(f'{i:5.3e} {j:5.3e} {k:5.3e} {l:5.3e} {m:5.3e} {n:5.3e}\n')
    f.close()

def transform_AMS02():
    R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/AMS-02_H_rigidity.txt')
    data = transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp)
    dump(data, 'AMS-02_H_energy.txt')
    
    R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/AMS-02_He_rigidity.txt')
    data = transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp, 2.)
    dump(data, 'AMS-02_He_energy.txt')

    R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp = compute_AMS02_CNO()
    data = transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp, 7.)
    dump(data, 'AMS-02_N_energy.txt')

    R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp = compute_AMS02_NeMgSi()
    data = transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp, 12.)
    dump(data, 'AMS-02_Mg_energy.txt')

    R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/AMS-02_Fe_rigidity.txt')
    data = transform_R2E(R_min, R_max, I_R, eStaLo, eStaUp, eSysLo, eSysUp, 26.)
    dump(data, 'AMS-02_Fe_energy.txt')

def transform_CALET():
    E_min, E_max, I_E, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/CALET_H_kEnergy.txt')
    data = [compute_mean_energy(E_min, E_max), I_E, eStaLo, eStaUp, eSysLo, eSysUp]
    dump(data, 'CALET_H_energy.txt')

    E_min, E_max, I_E, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/CALET_He_kEnergy.txt')
    data = [compute_mean_energy(E_min, E_max), I_E, eStaLo, eStaUp, eSysLo, eSysUp]
    dump(data, 'CALET_He_energy.txt')

    T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp = compute_CALET_CNO()
    data = transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, 14.)
    dump(data, 'CALET_N_energy.txt')

    T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/CALET_Fe_kEnergyPerNucleon.txt')
    data = transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, 56.)
    dump(data, 'CALET_Fe_energy.txt')
    
def transform_DAMPE():
    E_min, E_max, I_E, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/DAMPE_H_kEnergy.txt')
    data = [compute_mean_energy(E_min, E_max), I_E, eStaLo, eStaUp, eSysLo, eSysUp]
    dump(data, 'DAMPE_H_energy.txt')

    E_min, E_max, I_E, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/DAMPE_He_kEnergy.txt')
    data = [compute_mean_energy(E_min, E_max), I_E, eStaLo, eStaUp, eSysLo, eSysUp]
    dump(data, 'DAMPE_He_energy.txt')

def transform_CREAM():
    T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp = compute_CREAM_CNO()
    data = transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, 14.)
    dump(data, 'CREAM_N_energy.txt')

    T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp = compute_CREAM_NeMgSi()
    data = transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, 22.)
    dump(data, 'CREAM_Mg_energy.txt')

    T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp = readfile('lake/CREAM_Fe_kEnergyPerNucleon.txt')
    data = transform_T2E(T_min, T_max, I_T, eStaLo, eStaUp, eSysLo, eSysUp, 56.)
    dump(data, 'CREAM_Fe_energy.txt')

if __name__== "__main__":
    transform_AMS02()
    transform_CALET()
    transform_DAMPE()
    transform_CREAM()
