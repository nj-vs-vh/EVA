import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

def savefig(plt, plotname):
    print (plotname)
    plt.savefig(plotname)
    
def model(E, Z, params):
    I0, R0, alpha, Rb, dalpha, s = params
    R = E / float(Z)
    y = I0 * np.power(R / R0, -alpha)
    y /= np.power(1. + np.power(R / Rb, s), dalpha / s)
    return y
    
def plot_data(ax, filename, slope, norm, fmt, color, label, zorder=1):
    E, y, err_stat_lo, err_stat_up, err_sys_lo, err_sys_up = np.loadtxt(filename,usecols=(0,1,2,3,4,5),unpack=True)
    E /= norm
    y = norm * np.power(E, slope) * y
    y_err_lo = norm * np.power(E, slope) * err_stat_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * err_stat_up # np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color, label=label,
                capsize=3.5, markersize=6, elinewidth=1.8, capthick=1.8, zorder=zorder)
    y_err_lo = norm * np.power(E, slope) * err_sys_lo # np.sqrt(err_stat_lo**2. + err_sys_lo**2.)
    y_err_up = norm * np.power(E, slope) * err_sys_up # np.sqrt(err_stat_up**2. + err_sys_up**2.)
    ax.errorbar(E, y, yerr=[y_err_lo, y_err_up], fmt=fmt, markeredgecolor=color, color=color,
                capsize=3.5, markersize=6, elinewidth=1.8, capthick=1.8, zorder=zorder)
               
def plot_direct_H(params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e3, 1e6])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_yscale('log')
        ax.set_ylim([5e3, 18e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'output/AMS-02_H_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'output/CALET_H_energy.txt', 2.7, 1., 'o', 'tab:blue', 'CALET', 2)
    plot_data(ax, 'output/DAMPE_H_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

    E = np.logspace(3, 6, 1000)
    ax.plot(E, model(E, 1., params))
    
    ax.legend(fontsize=15, loc='lower left')
    savefig(plt, 'EVA_direct_H.pdf')

def plot_direct_He(params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e3, 1e6])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_yscale('log')
        ax.set_ylim([5e3, 25e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'output/AMS-02_He_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 1)
    plot_data(ax, 'output/CALET_He_energy.txt', 2.7, 1., 'o', 'tab:blue', 'CALET', 2)
    plot_data(ax, 'output/DAMPE_He_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)

    E = np.logspace(3, 6, 1000)
    ax.plot(E, model(E, 2., params))

    ax.fill_between([1e3, 2e3], 5e3, 25e3, color='tab:gray', alpha=0.3)
    
    ax.legend(fontsize=15, loc='best')
    savefig(plt, 'EVA_direct_He.pdf')

def plot_direct_N(params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e3, 1e6])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_yscale('log')
        ax.set_ylim([3e3, 9e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'output/CREAM_N_energy.txt', 2.7, 1., 'o', 'tab:green', 'CREAM', 1)
    plot_data(ax, 'output/CALET_N_energy.txt', 2.7, 1., 'o', 'tab:blue', 'CALET', 2)
#    plot_data(ax, 'output/DAMPE_He_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)
    plot_data(ax, 'output/AMS-02_N_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 4)

    E = np.logspace(3, 6, 1000)
    ax.plot(E, model(E, 7., params))

    ax.fill_between([1e3, 7e3], 3e3, 9e3, color='tab:gray', alpha=0.3)

    ax.legend(fontsize=15, loc='best')
    savefig(plt, 'EVA_direct_N.pdf')

def plot_direct_Mg(params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e3, 1e6])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_yscale('log')
        ax.set_ylim([1e3, 5e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'output/CREAM_Mg_energy.txt', 2.7, 1., 'o', 'tab:green', 'CREAM', 1)
#    plot_data(ax, 'output/CALET_N_energy.txt', 2.7, 1., 'o', 'tab:blue', 'CALET', 2)
##    plot_data(ax, 'output/DAMPE_He_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)
    plot_data(ax, 'output/AMS-02_Mg_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 4)

    E = np.logspace(3, 6, 1000)
    ax.plot(E, model(E, 12., params))

    ax.fill_between([1e3, 12e3], 1e3, 9e3, color='tab:gray', alpha=0.3)

    ax.legend(fontsize=15, loc='best')
    savefig(plt, 'EVA_direct_Mg.pdf')
    
def plot_direct_Fe(params):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e3, 1e6])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        #ax.set_yscale('log')
        ax.set_ylim([1e3, 6e3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)

    plot_data(ax, 'output/CREAM_Fe_energy.txt', 2.7, 1., 'o', 'tab:green', 'CREAM', 1)
    plot_data(ax, 'output/CALET_Fe_energy.txt', 2.7, 1., 'o', 'tab:blue', 'CALET', 2)
#    plot_data(ax, 'output/DAMPE_He_energy.txt', 2.7, 1., 'o', 'tab:red', 'DAMPE', 3)
    plot_data(ax, 'output/AMS-02_Fe_energy.txt', 2.7, 1., 'o', 'tab:gray', 'AMS-02', 4)

    E = np.logspace(3, 6, 1000)
    ax.plot(E, model(E, 26., params))

    ax.fill_between([1e3, 26e3], 1e3, 6e3, color='tab:gray', alpha=0.3)

    ax.legend(fontsize=15, loc='best')
    savefig(plt,'EVA_direct_Fe.pdf')
 
def plot_all(params_H, params_He, params_N, params_Mg, params_Fe):
    def set_axes(ax):
        ax.set_xlabel('E [GeV]')
        ax.set_xscale('log')
        ax.set_xlim([1e5, 1e9])
        ax.set_ylabel(r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax.set_yscale('log')
        ax.set_ylim([1e3, 1e5])
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    set_axes(ax)
    
    E = np.logspace(3, 10, 3000)
    H = model(E, 1., params_H)
    ax.plot(E, H, label='H', ls='--')
    
    He = model(E, 2., params_He)
    ax.plot(E, He, label='He', ls='--')
    
    N = model(E, 7., params_N)
    ax.plot(E, N, label='N', ls='--')
    
    Mg = model(E, 12., params_Mg)
    ax.plot(E, Mg, label='Mg', ls='--')
    
    Fe = model(E, 26., params_Fe)
    ax.plot(E, Fe, label='Fe', ls='--')

    ax.plot(E, H + He + N + Mg + Fe, label='all')

#kiss_tables/IceTop_allParticle_totalEnergy.txt
#kiss_tables/KASCADE-Grande_QGSJet-II-04_allParticle_totalEnergy.txt
#kiss_tables/KASCADE-Grande_QGSJet-II-2_allParticle_totalEnergy.txt
#kiss_tables/KASCADE-Grande_QGSJet-II-3_allParticle_totalEnergy.txt
#kiss_tables/KASCADE-Grande_SIBYLL-2.3_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2005_QGSJET-01_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2005_SIBYLL-2.1_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2011_EPOS-199_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2011_QGSJET-01_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2011_QGSJET-II-02_allParticle_totalEnergy.txt
#kiss_tables/KASCADE_2011_SIBYLL-2.1_allParticle_totalEnergy.txt
#kiss_tables/NUCLEON_allParticle_totalEnergy.txt
#kiss_tables/TALE_allParticle_totalEnergy.txt
#kiss_tables/TA_allParticle_totalEnergy.txt
#kiss_tables/TUNKA-133_allParticle_totalEnergy.txt
#kiss_tables/Tibet_QGSJET+HD_allParticle_totalEnergy.txt
#kiss_tables/Tibet_QGSJET+PD_allParticle_totalEnergy.txt
#kiss_tables/Tibet_SIBYLL+HD_allParticle_totalEnergy.txt

    plot_data(ax, 'kiss_tables/HAWC_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:green', 'HAWC', 1)
    plot_data(ax, 'kiss_tables/IceTop_QGSJet-II-04_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:orange', 'CREAM', 1)
    plot_data(ax, 'kiss_tables/IceTop_SIBYLL-2.1_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:red', 'CREAM', 1)
    plot_data(ax, 'kiss_tables/KASCADE_2011_SIBYLL-2.1_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:gray', 'CREAM', 1)
    plot_data(ax, 'kiss_tables/KASCADE_2011_QGSJET-II-02_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:gray', 'CREAM', 1)

    plot_data(ax, 'kiss_tables/Tibet_SIBYLL+HD_allParticle_totalEnergy.txt', 2.7, 1., 'o', 'tab:olive', 'CREAM', 1)

#kiss_tables/_allParticle_totalEnergy.txt

    #ax.legend(fontsize=15, loc='best')
    savefig(plt,'EVA_direct_all.pdf')

if __name__== "__main__":
    # costants
    R0, s = 1e3, 10. # costant
    # free parameters
    I_H, I_He, I_N, I_Mg, I_Fe = 10.09e3, 8.32e3, 4.47e3, 3.26e3, 3.36e3
    alpha_H, alpha_He, alpha_n = -0.110, -0.262, -0.16
    Rb, dalpha = 11e3, 0.21
    
    params_H = [I_H, R0, alpha_H, Rb, dalpha, s]
    plot_direct_H(params_H)
    
    params_He = [I_He, R0, alpha_He, Rb, dalpha, s]
    plot_direct_He(params_He)
    
    params_N = [I_N, R0, alpha_n, Rb, dalpha, s]
    plot_direct_N(params_N)
    
    params_Mg = [I_Mg, R0, alpha_n, Rb, dalpha, s]
    plot_direct_Mg(params_Mg)
    
    params_Fe = [I_Fe, R0, alpha_n, Rb, dalpha, s]
    plot_direct_Fe(params_Fe)

    plot_all(params_H, params_He, params_N, params_Mg, params_Fe)
