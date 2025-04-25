import pandas
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def smi2xyz(smi, output_path, file_name, dim_2_list):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) == 0:   # 若rdkit可以将二维结构三维化，则使用rdkit，可以保证原子构成的完整性；若不行，则直接将二维结构转为xyz，后续发现问题再重新计算
        AllChem.EmbedMolecule(mol)
        #AllChem.MMFFOptimizeMolecule(mol)   # 对三维结构进行简单优化，节省后续步骤时间
    if AllChem.EmbedMolecule(mol) == -1:
        dim_2_list.append(file_name)   # 记录未转化为三维的linker
    mol_path = os.path.join(output_path, 'mol.mol')
    Chem.MolToMolFile(mol, mol_path)
    xyz_name = file_name + '.xyz'
    with os.popen('cmd','w') as cmd:   # openbabel的python API不如命令行好用，故调用cmd用babel命令行完成转化
        cmd.write('cd %s\n'%output_path)
        cmd.write(f'obabel mol.mol -O {xyz_name}\n')   # --gen3D在进行转化时会将羧基上的H删去，导致计算结果不统一，
        cmd.close()
    os.remove(mol_path)   # .mol文件不需要，删去

