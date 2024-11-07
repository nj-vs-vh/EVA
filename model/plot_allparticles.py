import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import set_axes, plot_data, plot_data_statonly, savefig

XLABEL = r'E [GeV]'
YLABEL = r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'

def plot_galactic_all():
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e3, 3e6], ylim=[3e4, 6.4e4], xscale='log', yscale='linear')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'DAMPE_all_energy.txt', 2.7, 1, 'o', 'r', 'DAMPE (PRELIMINARY)', 9)
    plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 2.7, 1, 'o', 'tab:orange', 'GAMMA', 4)
    plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'tab:brown', 'HAWC', 5)
    plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 2.7, 1, 'o', 'tab:pink', 'NUCLEON', 6)

    plot_data(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'c', 'LHAASO', 8)

    E, I_all, I_all_err = np.loadtxt('output/galactic_all_fit.txt', usecols=(0,1,2), unpack=True)
    norm = 1.20
    ax.plot(E, norm * I_all, ls='--', zorder=10)
    ax.fill_between(E, norm * (I_all - I_all_err), norm * (I_all + I_all_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.legend(fontsize=16)

    savefig(fig, 'EVA_all_galactic.pdf')

def plot_galactic_individual():
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e3, 3e6], ylim=[3e3, 6.4e4], xscale='log', yscale='log')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'DAMPE_all_energy.txt', 2.7, 1, 'o', 'r', 'DAMPE (PRELIMINARY)', 9)
    plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 2.7, 1, 'o', 'tab:orange', 'GAMMA', 4)
    plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'tab:brown', 'HAWC', 5)
    plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 2.7, 1, 'o', 'tab:pink', 'NUCLEON', 6)

    plot_data(ax, 'LHAASO_QGSJET-II-04_all_energy.txt', 2.7, 1, 'o', 'c', 'LHAASO', 8)

    norm = 1.20
    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    ax.plot(E, norm * I, ls='--', color='tab:blue', zorder=10)
    ax.fill_between(E, norm * (I - I_err), norm * (I + I_err), color='tab:gray', alpha=0.15, zorder=0)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    ax.plot(E, norm * I, ls='--', color='tab:green', zorder=10)
    ax.fill_between(E, norm * (I - I_err), norm * (I + I_err), color='tab:gray', alpha=0.15, zorder=0)

    E, I_C, I_C_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    E, I_O, I_O_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    I = I_C + I_O
    I_err = I_C_err + I_O_err
    ax.plot(E, norm * I, ls='--', color='tab:orange', zorder=10)
    ax.fill_between(E, norm * (I - I_err), norm * (I + I_err), color='tab:gray', alpha=0.15, zorder=0)

    E, I_Mg, I_Mg_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    E, I_Si, I_Si_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    I = I_Mg + I_Si
    I_err = I_Mg_err + I_Si_err
    ax.plot(E, norm * I, ls='--', color='tab:olive', zorder=10)
    ax.fill_between(E, norm * (I - I_err), norm * (I + I_err), color='tab:gray', alpha=0.15, zorder=0)

    E, I, I_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)
    ax.plot(E, norm * I, ls='--', color='tab:red', zorder=10)
    ax.fill_between(E, norm * (I - I_err), norm * (I + I_err), color='tab:gray', alpha=0.15, zorder=0)

    ax.vlines(1.3e4, 1e1, 1e6, ls=':', lw=1, color='tab:blue')
    ax.vlines(1.3e4 * 26, 1e1, 1e6, ls=':', lw=1, color='tab:red')
    #ax.legend(fontsize=16)

    savefig(fig, 'EVA_individual_galactic.pdf')

def plot_galactic_lnA():
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    set_axes(ax, xlabel=XLABEL, ylabel=r'$\langle$ ln A $\rangle$', xlim =[3e3, 3e6], ylim=[1.0, 2.0], xscale='log', yscale='linear')
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    plot_data(ax, 'LHAASO_QGSJET-II-04_lnA_energy.txt', 0., 1, 'o', 'c', 'LHAASO', 8)

    E, lnA, lnA_err = np.loadtxt('output/galactic_lnA_fit.txt', usecols=(0,1,2), unpack=True)
    ax.plot(E, lnA, ls='--')
    #ax.fill_between(E, lnA_lo, lnA_up, color='tab:gray', alpha=0.2)

    ax.hlines(np.log(4), 1e2, 1e7, ls='--', color='tab:gray')
    ax.text(5e5, 1.42, 'He', color='tab:gray', fontsize=26)

    savefig(fig, 'EVA_lnA_galactic.pdf')

# def plot_allparticles_dataonly():
#     fig, ax = plt.subplots(figsize=(11.50, 8.5))
#     set_axes(ax, xlabel=XLABEL, ylabel=YLABEL, xlim=[3e4, 1e10], ylim=[5e5, 1e7], xscale='log', yscale='log')
#     #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

#     plot_data_statonly(ax, 'AUGER_all_energy.txt', 3., 1, 'o', 'r', 'Pierre Auger 2019', 19)
#     plot_data_statonly(ax, 'DAMPE_all_energy.txt', 3., 1, 'o', 'b', 'DAMPE', 20)
#     plot_data_statonly(ax, 'GAMMA_SIBYLL_all_energy.txt', 3., 1, 'o', 'tab:orange', 'GAMMA 2014')
#     plot_data_statonly(ax, 'HAWC_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'tab:brown', 'HAWC 2017')
#     plot_data_statonly(ax, 'ICECUBE_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'tab:gray', 'IceCube+IceTop 2019', 18)
#     plot_data_statonly(ax, 'ICETOP_QGSJET-II-04_all_energy.txt', 3., 1., 'o', 'tab:gray', 'IceTop', 1)
#     plot_data_statonly(ax, 'ICETOP_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'tab:green', 'IceTop SIBYLL 2020')
#     plot_data_statonly(ax, 'KASCADE_QGSJET-01_all_energy.txt', 3., 1, 'o', 'tab:red', 'KASCADE SIBYLL 2005')
#     plot_data_statonly(ax, 'KASCADE_SIBYLL_21_all_energy.txt', 3., 1, 'o', 'tab:red', 'KASCADE SIBYLL 2005')
#     plot_data_statonly(ax, 'KGRANDE_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'tab:olive', 'KASCADE-Grande QGSJET 2013')
#     plot_data_statonly(ax, 'KGRANDE_SIBYLL_23_all_energy.txt', 3., 1, 'o', 'tab:olive', 'KASCADE-Grande QGSJET 2013')
#     plot_data_statonly(ax, 'NUCLEON_all_energy.txt', 3., 1, 'o', 'tab:pink', 'NUCLEON 2019')
#     plot_data_statonly(ax, 'TALE_QGSJET-II-04_all_energy.txt', 3., 1, 'o', 'tab:purple', 'TALE 2018')
#     plot_data_statonly(ax, 'TA_all_energy.txt', 3., 1, 'o', 'b', 'Telescope Array 2015', 20)
#     plot_data_statonly(ax, 'TIBET_QGSJET+HD_all_energy.txt', 3., 1, 'o', 'tab:blue', 'Tibet-III QGSJET 2008')
#     plot_data_statonly(ax, 'TIBET_QGSJET+PD_all_energy.txt', 3., 1, 'o', 'tab:blue', 'Tibet-III QGSJET 2008')
#     plot_data_statonly(ax, 'TIBET_SIBYLL+HD_all_energy.txt', 3., 1, 'o', 'tab:blue', 'Tibet-III QGSJET 2008')
#     plot_data_statonly(ax, 'TUNKA-133_QGSJET-01_all_energy.txt', 3., 1, 'o', 'tab:cyan', 'TUNKA-133 2014')

#     plot_galactic(ax)

#     #ax.legend(fontsize=16)
#     savefig(fig, 'EVA_allaparticles_data.pdf')

if __name__== "__main__":
    plot_galactic_all()
    plot_galactic_lnA()
    plot_galactic_individual()