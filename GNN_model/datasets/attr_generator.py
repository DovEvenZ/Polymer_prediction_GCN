'''Attribute generator in gnn_attribute.
'''
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from datasets.utils import one_hot


def get_node_attr(mol,
                  node_attr_mat: list,
                  atom_type: dict,
                  filter: list,
                  ):
    '''Generator of node attributes.

    Args:
        mol: rdkit mol object, has N atoms.
        node_attr_mat: (multiple) extra attributes need to add to the nodes.
            Maybe in form of np.ndarray (N atoms * M attributes),
            or a list (length=M) of list (length=N).
        atom_type: dict of element symbol and index in one hot repr.
        filter: list of attr index that need to be filtered.

    Return:
        mol_node_attr(np.array, shape=N*(len(_ATOM_TYPE)+4+X)): including
            one hot repr of atom type,
            atomic number,
            whether is aromatic,
            number of atom neighbors,
            number of Hs in neighbors,
            extra attributes (if exists).
        pos(np.array, shape=N*3):
            atom coord.
    '''
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    type_idx_hot = []
    atomic_number = []
    aromatic = []
    num_neighbors = []
    num_hs = []

    for atom in mol.GetAtoms():
        type_idx = atom_type[atom.GetSymbol()]
        type_idx_hot.append(one_hot(type_idx, len(atom_type)))
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        neighbors = atom.GetNeighbors()
        num_neighbors.append(len(neighbors))
        nhs = 0
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 1:
                nhs += 1
        num_hs.append(nhs)

    if node_attr_mat:
        mol_node_attr = np.concatenate((np.array(type_idx_hot),
                                        np.array(atomic_number)[:, np.newaxis],
                                        np.array(aromatic)[:, np.newaxis],
                                        np.array(num_neighbors)[:, np.newaxis],
                                        np.array(num_hs)[:, np.newaxis],
                                        np.array(node_attr_mat).T,
                                        ), axis=1, dtype=float)
    else:
        mol_node_attr = np.concatenate((np.array(type_idx_hot),
                                        np.array(atomic_number)[:, np.newaxis],
                                        np.array(aromatic)[:, np.newaxis],
                                        np.array(num_neighbors)[:, np.newaxis],
                                        np.array(num_hs)[:, np.newaxis],
                                        ), axis=1, dtype=float)
    # add filter
    node_filter = [1] * mol_node_attr.shape[1]
    for nfi in filter:
        node_filter[nfi] = 0
    node_filter = np.array(node_filter, dtype=bool)
    mol_node_attr = mol_node_attr[:, node_filter]

    return torch.tensor(mol_node_attr, dtype=torch.float), pos


def get_adj_mat(mol, tot_atom: int=0):
    '''Generator of adj matrix.

    Args:
        mol: rdkit mol object, has M bonds.
        tot_atom: number of atoms already in this graph.
        
    Return:
        mol_adj_mat(np.array, shape=2M*2): adj matrix of this mol.
    '''
    start_list = []
    end_list = []
    for bond in mol.GetBonds():
        start_list.append(bond.GetBeginAtomIdx())
        end_list.append(bond.GetEndAtomIdx())

        start_list.append(bond.GetEndAtomIdx())
        end_list.append(bond.GetBeginAtomIdx())
    
    mol_adj_mat = torch.tensor([start_list,
                                end_list])

    return mol_adj_mat


def get_edge_type(mol,
                  bond_type: dict,
                  filter: list,
                  extra_attr: None,
                  ):
    '''Generator of edge type.

    Args:
        mol: rdkit mol object, has M bonds.
        bond_type: dict of bond type in rdkit(key) to one hot index(value).
        filter: list of attr index that need to be filtered.
        extra_attr: (multiple) extra attributes need to add to the edges.
            Maybe in form of np.ndarray (N bonds * M attributes),
            or a list (length=M) of list (length=N).
        
    Return:
        mol_edge_type(torch.tensor, shape=2M*len(_BOND_TYPE)): including
            one hot repr of bond type,
            extra attributes (if exists).
            # remember to double the extra attributes (twice for each bond)
    '''
    edge_type = []
    for bond in mol.GetBonds():
        edge_type += 2 * [bond_type[bond.GetBondType()]]
    
    mol_edge_type = torch.tensor([one_hot(t, len(bond_type)) for t in edge_type],
                                 dtype=torch.float)
    # add filter
    edge_filter = [1] * mol_edge_type.shape[1]
    for efi in filter:
        edge_filter[efi] = 0
    edge_filter = np.array(edge_filter, dtype=bool)
    mol_edge_type = mol_edge_type[:, edge_filter]

    return mol_edge_type
