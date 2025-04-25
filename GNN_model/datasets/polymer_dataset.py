import json
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import (Data,
                                  InMemoryDataset,
                                  )

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
RDLogger.DisableLog('rdApp.*')

from datasets.attr_generator import (get_adj_mat,
                                     get_edge_type,
                                     get_node_attr,
                                     )


# definition atom and bond type for one hot repr
_ATOM_TYPE = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'B': 8, 'Si': 9, 'P': 10, 'Na': 11, 'I': 12}
_BOND_TYPE = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


class Polymer(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None,
                 sdf_file: str=None,
                 node_attr_file: str=None,
                 edge_attr_file: str=None,
                 graph_attr_file: str=None,
                 node_attr_list: List[str]=[],
                 edge_attr_list: List[str]=[],
                 graph_attr_list: List[str]=[],
                 target_list: List[str]=[],
                 node_attr_filter: Union[int, List]=[],
                 edge_attr_filter: Union[int, List]=[],
                 reprocess: bool=False,
                 ):
        '''graph dataset for polymers

        Args:
            root: base dir to store raw and processed data,
            transform: data transformer applied when generating graph(default to None),
            pre_transform: inherited from InMemoryDataset, not used(default to None),
            pre_filter: inherited from InMemoryDataset, not used(default to None),
            sdf_file: path to sdf file containing all molecules 3D structures (MUST HAVE),
            node_attr_file: path to json file containing all extra node attributes,
            edge_attr_file: path to json file containing all extra edge attributes,
            graph_attr_file: path to csv file containing all target(s) and graph_attrs (MUST HAVE),
            node_attr_list: list containing node attr(s) demanding consideration,
            edge_attr_list: list containing edge attr(s) demanding consideration,
            graph_attr_list: list containing graph attr(s) demanding consideration,
            target_list: list containing the target(s),
            node_attr_filter: index of node attr(s) to be filtered(default to []),
            edge_attr_filter: index of edge attr(s) to be filtered(default to []),
            reprocess: whether to force reprocess the dataset.

        Return:
            A InMemoryDataset of polymer graph data.
        '''
        self.sdf_file = sdf_file
        self.node_attr_file = node_attr_file
        self.edge_attr_file = edge_attr_file
        self.graph_attr_file = graph_attr_file
        self.node_attr_list = node_attr_list
        self.edge_attr_list = edge_attr_list
        self.graph_attr_list = graph_attr_list
        self.target_list = target_list
        self.node_attr_filter = [node_attr_filter] if isinstance(node_attr_filter, int) else node_attr_filter  # TODO: Implement this
        self.edge_attr_filter = [edge_attr_filter] if isinstance(edge_attr_filter, int) else edge_attr_filter  # TODO: Implement this
        self.reprocess = reprocess  # TODO: Implement this
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        '''calculate mean value of all targets.
        '''
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        '''calculate standard deviation value of all targets.
        '''
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())
    
    @property
    def raw_file_names(self) -> List[str]:
        '''define default path to input data files.
        '''
        return ['merged_mol.sdf', 'smiles.csv']
    
    @property
    def processed_file_names(self) -> str:
        '''define default path to processed data file.
        '''
        return 'polymer_data_v1.pt'
    
    @property
    def num_graph_features(self) -> int:
        '''number of graph attributes.
        '''
        return len(self.graph_attr_list)
    
    @property
    def num_total_features(self) -> int:
        '''number of all(node, edge, graph) attributes.
        '''
        return (self.num_node_features +
                self.num_edge_features +
                self.num_graph_features)

    def process(self):
        '''process raw data to generate dataset
        '''
        # read graph_attr_file
        database = pd.read_csv(self.graph_attr_file)

        # extract target and graph_attr from csv
        target = torch.tensor(
            np.array(database.loc[:, self.target_list]),
            dtype = torch.float
            ).reshape(-1,len(self.target_list))
        if not self.graph_attr_list == []:
            graph_attr = torch.tensor(
                np.array(database.loc[:, self.graph_attr_list]),
                dtype=torch.float
                ).reshape(-1,len(self.graph_attr_list))

        # read sdf file
        suppl = Chem.SDMolSupplier(self.sdf_file,
                                   removeHs=False,
                                   sanitize=False,
                                   )

        # read node attribute
        if self.node_attr_file is None:
            node_attr_dict = None
        else:
            with open(self.node_attr_file) as nf:
                node_attr_dict = json.load(nf)
            length_list = [len(value) for value in node_attr_dict.values()]
            assert min(length_list) == max(length_list), 'Node attributes are not of the same length.'
            assert min(length_list) == len(suppl), 'Node attributes and SDF file are not of the same length.'

        # extract raw attrs from structures and generate graph
        data_list = []
        for i, mol in enumerate(suppl):
            # node attr
            node_attr_mat = []
            for node_attr in self.node_attr_list:
                node_attr_mat.append(node_attr_dict[node_attr][i])   # num_node_attrs * num_atoms
            x, pos = get_node_attr(mol,
                                   node_attr_mat,
                                   _ATOM_TYPE,
                                   self.node_attr_filter,
                                   )

            # edge attr
            edge_attr = get_edge_type(mol,
                                      _BOND_TYPE,
                                      self.edge_attr_filter,
                                      None,
                                      )
            # edge index(adj matrix)
            edge_index = get_adj_mat(mol)
            # target
            y = target[i].unsqueeze(0)
            # graph name
            name = mol.GetProp('_Name')
            # graph attr
            if not self.graph_attr_list == []:
                g_a = graph_attr[i].unsqueeze(0)
            # create mol graph
                data = Data(x=x, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=y, graph_attr=g_a,
                            name=name, idx=i)
            else:
                data = Data(x=x, pos=pos, edge_index=edge_index,
                            edge_attr=edge_attr, y=y,
                            name=name, idx=i)
            # pre filter and pre transform
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # graph generated, add to data list
            data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])
