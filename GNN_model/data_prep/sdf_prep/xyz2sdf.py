'''Convert bunch of xyz files' first structures
into a single sdf file.
'''
import os
import subprocess
from typing import List
from rdkit import Chem


def xyz2sdf(xyzfile):
    sdffile = xyzfile.replace('.xyz', '.sdf')
    openbabel_cmd = f'obabel -ixyz {xyzfile} -osdf -O {sdffile}'

    subprocess.Popen(openbabel_cmd,
                     shell=True,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)

    return sdffile


def merge_sdf(sdf_list: List,
              output_path: str,
              need_remove: bool=False):
    
    print(f'total sdf: {len(sdf_list)}')
    
    mol_list = []
    name_list = []
    for sdf in sdf_list:
        mol_name = os.path.basename(sdf).split('.')[0]
        mol = Chem.SDMolSupplier(sdf, removeHs=False,
                                 sanitize=False)[0]
        mol_list.append(mol)
        name_list.append(mol_name)
    
    out_sdf_name = output_path
    writer = Chem.SDWriter(out_sdf_name)
    for i, mol in enumerate(mol_list):
        mol.SetProp('_Name', name_list[i])
        writer.write(mol)
    
    writer.close()

    if need_remove:
        for f in sdf_list:
            os.remove(f)

    return out_sdf_name
