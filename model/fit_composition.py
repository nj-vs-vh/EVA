import numpy as np
from iminuit import Minuit
from jacobi import propagate

def model(E, Z, params):
    I0, R0, alpha, Rb, dalpha, s = params
    R = E / float(Z)
    y = I0 * np.power(R / R0, -alpha)
    y /= np.power(1. + np.power(R / Rb, s), dalpha / s)
    return y

def load_data(filename, slope, norm, minEnergy, maxEnergy = 1e20):
    filename = 'data/' + filename
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E = E * norm
    y = np.power(E, slope) * y / norm
    y_err_lo = np.power(E, slope) * np.sqrt(err_stat_lo**2. + err_sys_lo**2.) / norm
    y_err_up = np.power(E, slope) * np.sqrt(err_stat_up**2. + err_sys_up**2.) / norm
    items = [i for i in range(len(E)) if (E[i] > minEnergy and E[i] < maxEnergy)]
    return E[items], y[items], y_err_lo[items], y_err_up[items]

def experiment_chi2(filename, Z, par, norm = 1.):
    xd, yd, errd_lo, errd_up = load_data(filename, 2.7, norm, 1e3 * Z)
    chi2 = 0.
    for x_i, y_i, err_lo_i, err_up_i in zip(xd, yd, errd_lo, errd_up):
        m = model(x_i, Z, par)
        if m > y_i:
            chi2 += np.power((m - y_i) / err_up_i, 2.)
        else:
            chi2 += np.power((m - y_i) / err_lo_i, 2.)
    return chi2

def fit(params):
    def chi2_function(I_H, I_He, I_N, I_Mg, I_Fe, R0, alpha_H, alpha_He, alpha_n, Rb, dalpha, s, fCREAM):
        chi2 = 0.
        # H
        chi2 += experiment_chi2('AMS-02_H_energy.txt', 1., [I_H, R0, alpha_H, Rb, dalpha, s])
        chi2 += experiment_chi2('CALET_H_energy.txt', 1., [I_H, R0, alpha_H, Rb, dalpha, s])
        chi2 += experiment_chi2('DAMPE_H_energy.txt', 1., [I_H, R0, alpha_H, Rb, dalpha, s])
        # He
        chi2 += experiment_chi2('AMS-02_He_energy.txt', 2., [I_He, R0, alpha_He, Rb, dalpha, s])
        chi2 += experiment_chi2('CALET_He_energy.txt', 2., [I_He, R0, alpha_He, Rb, dalpha, s])
        chi2 += experiment_chi2('DAMPE_He_energy.txt', 2., [I_He, R0, alpha_He, Rb, dalpha, s])
        # N
        chi2 += experiment_chi2('AMS-02_N_energy.txt', 7., [I_N, R0, alpha_n, Rb, dalpha, s])
        chi2 += experiment_chi2('CALET_N_energy.txt', 7., [I_N, R0, alpha_n, Rb, dalpha, s])
        chi2 += experiment_chi2('CREAM_N_energy.txt', 7., [I_N, R0, alpha_n, Rb, dalpha, s], fCREAM)
        # Mg
        chi2 += experiment_chi2('AMS-02_Mg_energy.txt', 12., [I_Mg, R0, alpha_n, Rb, dalpha, s])
        chi2 += experiment_chi2('CREAM_Mg_energy.txt', 12., [I_Mg, R0, alpha_n, Rb, dalpha, s], fCREAM)
        # Fe
        chi2 += experiment_chi2('AMS-02_Fe_energy.txt', 26., [I_Fe, R0, alpha_n, Rb, dalpha, s])
        chi2 += experiment_chi2('CALET_Fe_energy.txt', 26., [I_Fe, R0, alpha_n, Rb, dalpha, s])
        return chi2

    I_H, I_He, I_N, I_Mg, I_Fe, R0, alpha_H, alpha_He, alpha_n, Rb, dalpha, s = params

    m = Minuit(chi2_function,
               I_H=I_H, I_He=I_He, I_N=I_N, I_Mg=I_Mg, I_Fe=I_Fe,
               R0=R0,
               alpha_H=alpha_H, alpha_He=alpha_He, alpha_n=alpha_n,
               Rb=Rb, dalpha=dalpha, s=s,
               fCREAM=1.)

    m.limits['fCREAM'] = (0.8, 1.2)

    m.fixed['R0'] = True
    m.fixed['s'] = True

    m.errordef = Minuit.LEAST_SQUARES

    m.simplex()
    m.migrad()
    m.hesse()

    print(m)

    return m.values, m.errors, m.fval
 
if __name__== "__main__":
    # costants
    R0, s = 1e3, 5. # costant
    # free parameters
    I_H, I_He, I_N, I_Mg, I_Fe = 10.09e3, 8.32e3, 4.47e3, 3.26e3, 3.36e3
    alpha_H, alpha_He, alpha_n = -0.110, -0.262, -0.16
    Rb, dalpha = 11e3, 0.21

    params = [I_H, I_He, I_N, I_Mg, I_Fe, R0, alpha_H, alpha_He, alpha_n, Rb, dalpha, s]
    values, errors, fval = fit(params)

    print(fval)
