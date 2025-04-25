from typing import List

def get_disp(disp_file: str) -> List:
    disp_list = []
    with open(disp_file) as df:
        text = df.readlines()
    num_atoms = int(text[0].split()[-2])
    for i in range(1, 1 + num_atoms):
        info = text[i][:-3]
        disp_E = float(info.split()[-2])
        disp_list.append(disp_E)
    return disp_list

def get_charge(chg_file: str) -> List:
    charge_list = []
    with open(chg_file) as cf:
        text = cf.readlines()
    for line in text:
        charge_list.append(float(line.split()[-1]))
    return charge_list
