import crdb

def print_column_names(tab):
    for icol, col_name in enumerate(tab.dtype.fields):
        print("%2i" % icol, col_name)

def dump_datafile(quantity, energyType, expName, subExpName, filename, combo_level=0):
    filename = 'lake/' + filename
    print(f"search for {quantity} as a function of {energyType} measured by {expName}")
    
    tab = crdb.query(quantity, energy_type=energyType, combo_level=combo_level, energy_convert_level=0, exp_dates=expName)
 
    subExpNames = set(tab["sub_exp"])
    print("number of datasets found : ", len(subExpNames))
    print(subExpNames)

    adsCodes = set(tab["ads"])
    print(adsCodes)

    items = [i for i in range(len(tab["sub_exp"])) if tab["sub_exp"][i] == subExpName]
    print("number of data : ", len(items))
    assert(len(items) > 0)

    print(f"dump on {filename}")
    with open(filename, 'w') as f:
        f.write(f"#source: CRDB\n")
        f.write(f"#Quantity: {quantity}\n")
        f.write(f"#EnergyType: {energyType}\n")
        f.write(f"#Experiment: {expName}\n")
        f.write(f"#ads: {tab['ads'][items[0]]}\n")
        f.write(f"#E_lo - E_up - y - errSta_lo - errSta_up - errSys_lo - errSys_up\n")
        for eBin, value, errSta, errSys in zip(tab["e_bin"][items], tab["value"][items], tab["err_sta"][items], tab["err_sys"][items]):
            f.write(f"{eBin[0]:10.5e} {eBin[1]:10.5e} {value:10.5e} {errSta[0]:10.5e} {errSta[1]:10.5e} {errSys[0]:10.5e} {errSys[1]:10.5e}\n")
    print("")

if __name__== "__main__":
#    dump_datafile("H", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_H_rigidity.txt")
#    dump_datafile("He", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_He_rigidity.txt")
#    dump_datafile("C", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_C_rigidity.txt")
#    dump_datafile("N", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_N_rigidity.txt")
#    dump_datafile("O", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_O_rigidity.txt")
#    dump_datafile("Ne", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_Ne_rigidity.txt")
#    dump_datafile("Mg", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_Mg_rigidity.txt")
#    dump_datafile("Si", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_Si_rigidity.txt")
#    dump_datafile("Fe", "R", "AMS02", "AMS02 (2011/05-2019/10)", "AMS-02_Fe_rigidity.txt")

#    dump_datafile("H", "Ekn", "CALET", "CALET (2015/10-2021/12)", "CALET_H_kEnergy.txt")
#    dump_datafile("He", "Ek", "CALET", "CALET (2015/10-2022/04)", "CALET_He_kEnergy.txt")
#    dump_datafile("C", "Ekn", "CALET", "CALET (2015/10-2019/10)", "CALET_C_kEnergyPerNucleon.txt")
#    dump_datafile("O", "Ekn", "CALET", "CALET (2015/10-2019/10)", "CALET_O_kEnergyPerNucleon.txt")
#    dump_datafile("Fe", "Ekn", "CALET", "CALET (2016/01-2020/05)", "CALET_Fe_kEnergyPerNucleon.txt")
#
#    dump_datafile("H", "Ek", "DAMPE", "DAMPE (2016/01-2018/06)", "DAMPE_H_kEnergy.txt")
#    dump_datafile("He", "Ek", "DAMPE", "DAMPE (2016/01-2020/06)", "DAMPE_He_kEnergy.txt")

    dump_datafile("C", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_C_kEnergyPerNucleon.txt")
    dump_datafile("O", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_O_kEnergyPerNucleon.txt")

    dump_datafile("Ne", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Ne_kEnergyPerNucleon.txt")
    dump_datafile("Mg", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Mg_kEnergyPerNucleon.txt")
    dump_datafile("Si", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Si_kEnergyPerNucleon.txt")

    dump_datafile("Fe", "Ekn", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Fe_kEnergyPerNucleon.txt")
    
    
