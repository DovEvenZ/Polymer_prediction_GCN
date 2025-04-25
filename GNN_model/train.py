import os
# 禁用BLAS库的多线程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
    
import yaml
import random
import time
import math
import json
import shutil
import numpy as np

import torch
import torch.nn.functional as F

# 设置PyTorch使用的线程数为1
torch.set_num_threads(1)
# 设置PyTorch内部操作之间的并行度为1
torch.set_num_interop_threads(1)

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops, to_networkx
from torch.utils.data import Subset, random_split

# tensorboard
from torch.utils.tensorboard import SummaryWriter
# dataset
from datasets.polymer_dataset import Polymer
# net
from nets.mpnn_basic import mpnn_basic
from nets.readout_add_graph_feature import readout_add_graph_feature

from utils.calc_mean_std import calc_mean_std
from utils.plot import pred_tgt, loss_epoch
from utils.time import convert_time
from utils.post_processing import read_log

from utils.change_yaml_KC import change_yaml_KC
from utils.KF_random import KF_random
from utils.graph_hyperparameter import graph_hyperparameter
from sklearn.model_selection import KFold

class CustomSubset(Subset):
    """A custom subset class that retains the 'graph_attr' and 'y' attributes."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.graph_attr = torch.as_tensor(
                [dataset.graph_attr[i].tolist() for i in indices]
            )  # Retain the 'graph_attr' attribute
        self.y = torch.as_tensor(
                [dataset.y[i].tolist() for i in indices]
            )  # Retain the 'y' attribute

    def __getitem__(self, idx):
        # Support indexing
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self):
        # Support len()
        return len(self.indices)

def main():
    with open('model_parameters.yml') as mp:
        param = yaml.full_load(mp)

    path_raw = param['path']
    jobtype_raw = param['jobtype']
    mode = param['mode']

    path = f'{path_raw}/{jobtype_raw}'
    jobtype = f'{jobtype_raw}___{mode}'

    SDF_FILE = param['sdf_file']
    NODE_ATTR_FILE = param['node_attr_file']
    EDGE_ATTR_FILE = param['edge_attr_file']
    GRAPH_ATTR_FILE = param['graph_attr_file']
    node_attr_list = param['node_attr_list']
    edge_attr_list = param['edge_attr_list']
    graph_attr_list = param['graph_attr_list']
    target_list = param['target_list']
    node_attr_filter = param['node_attr_filter']
    edge_attr_filter = param['edge_attr_filter']

    if os.path.exists(os.path.join(path, 'processed/')):
        shutil.rmtree(os.path.join(path, 'processed/'))

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed = param['seed']
    setup_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    batch_size = param['batch_size']

    dim_linear = param['dim_linear']
    dim_conv = param['dim_conv']
    processing_steps = param['processing_steps']
    mp_times = param['mp_times']

    epoch_num = param['epoch_num']

    Distance = param['Distance']

    if 'K' in mode:
        KC = True
    else:
        KC = False

    LOG_DIR = f'./log/{hash(time.time())}'  # logdir for tensorboard
    writer = SummaryWriter(LOG_DIR)

    class MyTransform(object):
        def __call__(self, data):
            # Specify target.
            data.y = data.y[:, 0]  # only one target
            return data

    class Complete(object):
        def __call__(self, data):
            device = data.edge_index.device

            row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
            col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

            row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
            col = col.repeat(data.num_nodes)
            edge_index = torch.stack([row, col], dim=0)

            edge_attr = None
            if data.edge_attr is not None:
                idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                size = list(data.edge_attr.size())
                size[0] = data.num_nodes * data.num_nodes
                edge_attr = data.edge_attr.new_zeros(size)
                edge_attr[idx] = data.edge_attr

            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data.edge_attr = edge_attr
            data.edge_index = edge_index

            return data

    if Distance:
        transform = T.Compose([MyTransform(), Complete()
                                ,T.Distance(norm=False)
                                ])        
    else:
        transform = T.Compose([MyTransform(), Complete()
                                #,T.Distance(norm=False)
                                ])

    dataset = Polymer(path,
                    transform=transform,
                    sdf_file=SDF_FILE,
                    node_attr_file=NODE_ATTR_FILE,
                    edge_attr_file=EDGE_ATTR_FILE,
                    graph_attr_file=GRAPH_ATTR_FILE,
                    node_attr_list=node_attr_list,
                    edge_attr_list=edge_attr_list,
                    graph_attr_list=graph_attr_list,
                    target_list=target_list,
                    node_attr_filter=node_attr_filter,
                    edge_attr_filter=edge_attr_filter,
                    reprocess=True,
                    )

    if param['split_method'] == 'random':
        # Split datasets.
        train_size = int(param['train_size'] * len(dataset))
        val_size = int(param['val_size'] * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # 设置seed，保证每次划分的数据相同
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
        train_dataset = CustomSubset(dataset, train_dataset.indices)
        # 获取训练集的平均值和标准差
        mean, std = calc_mean_std(
            torch.cat(
                [
                    train_dataset.graph_attr,
                    train_dataset.y
                    ],
                dim = 1
                )
            )

        data_scaled = (torch.cat([
                        dataset.graph_attr, 
                        dataset.y], dim = 1) - mean) / std

        dataset.data.graph_attr = data_scaled[:, 0:len(graph_attr_list)]
        dataset.data.y = data_scaled[:, len(graph_attr_list):len(graph_attr_list) + len(target_list)]

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    elif param['split_method'] == 'manual':
        split_path = param['SPLIT_file']
        train_dataset = dataset[np.load(split_path, allow_pickle=True)[0]]
        # 获取训练集的平均值和标准差
        mean, std = calc_mean_std(
            torch.cat(
                [
                    train_dataset.graph_attr,
                    train_dataset.y
                    ],
                dim = 1
                )
            )

        data_scaled = (torch.cat([dataset.graph_attr, dataset.y], dim = 1) - mean) / std

        dataset.data.graph_attr = data_scaled[:, 0:len(graph_attr_list)]
        dataset.data.y = data_scaled[:, len(graph_attr_list):len(graph_attr_list) + len(target_list)]

        train_dataset = dataset[np.load(split_path, allow_pickle=True)[0]]
        val_dataset = dataset[np.load(split_path, allow_pickle=True)[1]]
        test_dataset = dataset[np.load(split_path, allow_pickle=True)[2]]

    else:
        raise NotImplementedError("Split method not implemented.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    net = readout_add_graph_feature(dataset, dim_linear, dim_conv, processing_steps, mp_times)  # with graph_attr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    if param['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = param['lr'])
    elif param['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr = param['lr'])
    elif param['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = param['lr'])
    else:
        raise NotImplementedError("Optimizer not implemented.")

    scheduler = None
    if param['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=param['scheduler_factor'], patience=param['scheduler_patience'],
            min_lr=param['scheduler_min_lr']
            )
    elif param['scheduler'] == 'Exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=param['scheduler_factor']
            # min_lr=param['scheduler_min_lr']
            )
    else:
        print('No scheduler')
        scheduler = None

    def train(epoch):
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_loader.dataset)

    def test(loader):
        model.eval()
        error_mae = 0
        error_mse = 0
        error_s2 = 0

        for data in loader:
            data = data.to(device)
            error_mae += (model(data) * std[-1] - data.y * std[-1]).abs().sum().item()  # MAE
            error_mse += ((model(data) * std[-1] - data.y * std[-1]) ** 2).sum().item()  # MSE
            error_s2 += ((data.y * std[-1]) ** 2).sum().item()

        MAE = error_mae / len(loader.dataset)
        R2 = 1 - error_mse / error_s2
        return MAE, R2

    best_MAE_val = math.inf
    best_epoch = None
    _best_MAE_test = None

    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    os.makedirs(f'Training_Recording/{jobtype}/{TIME}')
    shutil.copy('model_parameters.yml', f'Training_Recording/{jobtype}/{TIME}')
    os.makedirs(f'Training_Recording/{jobtype}/{TIME}/Plot')
    os.makedirs(f'Training_Recording/{jobtype}/{TIME}/Model')
    log_file = f'Training_Recording/{jobtype}/{TIME}/training_{TIME}.log'
    with open(log_file, 'a') as tl:
        data_abspath = os.path.abspath(param['path'])
        tl.write(f'data_path: {data_abspath}\n')
        tl.write(f'{json.dumps(param)}\n')
        tl.write(f'size of test set: {len(test_loader)}\n')
        tl.write(f'size of val set: {len(val_loader)}\n')
        tl.write(f'size of training set: {len(train_loader)}\n')
        tl.write("Begin training...\n")

    data_dict = {'Train': {}, 'Val': {}, 'Test': {}}
    data_file = f'Training_Recording/{jobtype}/{TIME}/data_name_{TIME}.log'

    for phase, check_dataset in zip(['Train', 'Val', 'Test'], [train_dataset, val_dataset, test_dataset]):
        data_dict[phase] = check_dataset
    with open(data_file,'a') as data_f:
        for phase in ['Train', 'Val', 'Test']:
            data_f.write(f'{phase}:\n')
            for data_i in data_dict[phase]:
                data_f.write(f'{data_i.name}\n')

    start_time = time.perf_counter()
    for epoch in range(1, epoch_num+1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        MAE_train, R2_train = test(train_loader)
        if not KC:
            MAE_val, R2_val = test(val_loader)
        else:
            MAE_val, R2_val = (0,0)
        MAE_test, R2_test = test(test_loader)
        if KC:
            if epoch % param['scheduler_patience'] == 0:
                lr = scheduler.get_last_lr()[0]
                if lr > param['scheduler_min_lr']:
                    scheduler.step()
                else:
                    scheduler.optimizer.param_groups[0]['lr'] = param['scheduler_min_lr']

        else:
            scheduler.step(MAE_val)

            if MAE_val <= best_MAE_val:
                model.eval()
                best_MAE_val = MAE_val
                best_epoch = epoch
                _best_MAE_test = MAE_test
                torch.save(model.state_dict(), f'Training_Recording/{jobtype}/{TIME}/Model/best_model_{TIME}.pt')
                torch.save(model, f'Training_Recording/{jobtype}/{TIME}/Model/best_model_{TIME}.pkl')
                sum_val_x = []
                sum_val_y = []
                for data in val_loader:
                    data = data.to(device)
                    sum_val_y.append(model(data))
                    sum_val_x.append(data.y)
                pred_tgt(
                    torch.cat(sum_val_x) * std[-1] + mean[-1],
                    torch.cat(sum_val_y) * std[-1] + mean[-1],
                    MAE_val,
                    R2_val,
                    f'Training_Recording/{jobtype}/{TIME}/Plot/best_model_val_{TIME}.png'
                    )
                sum_test_x = []
                sum_test_y = []
                for data in test_loader:
                    data = data.to(device)
                    sum_test_y.append(model(data))
                    sum_test_x.append(data.y)
                pred_tgt(
                    torch.cat(sum_test_x) * std[-1] + mean[-1],
                    torch.cat(sum_test_y) * std[-1] + mean[-1],
                    MAE_test,
                    R2_test,
                    f'Training_Recording/{jobtype}/{TIME}/Plot/best_model_test_{TIME}.png'
                    )
        '''
        writer.add_scalar('Learning rate', lr, epoch)
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('MAE/Validation', MAE_val, epoch)
        writer.add_scalar('MAE/Test', MAE_test, epoch)
        '''
        if epoch % param['output_step'] == 0:
            with open(log_file, 'a') as tl:
                tl.write(
                    f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                    f'Train MAE: {MAE_train:.7f}, Train R2: {R2_train:.7f}, '
                    f'Val MAE: {MAE_val:.7f}, Val R2: {R2_val:.7f}, '
                    f'Test MAE: {MAE_test:.7f}, Test R2: {R2_test:.7f}'
                    '\n'
                    )
    end_time = time.perf_counter()
    tot_time = end_time - start_time
    epoch_time = tot_time / epoch_num

    with open(log_file, 'a') as tl:
        if not KC:
            tl.write(f'Best_MAE_val: {best_MAE_val}\n')
            tl.write(f'Best_epoch: {best_epoch}\n')
            tl.write(f'Corresponding MAE_test: {_best_MAE_test}\n')
        hours, minutes, seconds = convert_time(tot_time)
        tl.write(f'Total_time: {hours} h {minutes} m {seconds} s\n')
        hours, minutes, seconds = convert_time(epoch_time)
        tl.write(f'Time_per_epoch: {hours} h {minutes} m {seconds} s\n')

    model.eval()
    torch.save(model.state_dict(), f'Training_Recording/{jobtype}/{TIME}/Model/last_model_{TIME}.pt')
    torch.save(model, f'Training_Recording/{jobtype}/{TIME}/Model/last_model_{TIME}.pkl')
    sum_val_x = []
    sum_val_y = []

    if not KC:
        for data in val_loader:
            data = data.to(device)
            sum_val_y.append(model(data))
            sum_val_x.append(data.y)
        pred_tgt(
            torch.cat(sum_val_x) * std[-1] + mean[-1],
            torch.cat(sum_val_y) * std[-1] + mean[-1],
            MAE_val,
            R2_val,
            f'Training_Recording/{jobtype}/{TIME}/Plot/last_model_val_{TIME}.png'
            )

    log_info_dict = read_log(log_file).get_performance()
    epoch_list = log_info_dict['epoch']
    loss_list = log_info_dict['loss']
    train_MAE_list = log_info_dict['train_MAE']
    if not KC:
        val_MAE_list = log_info_dict['val_MAE']
    test_MAE_list = log_info_dict['test_MAE']
    loss_epoch([[epoch_list, loss_list]], ['loss-epoch'], ['#03658C'], 'epoch', 'loss', f'Training_Recording/{jobtype}/{TIME}/Plot/loss-epoch_{TIME}.png')
    loss_epoch([[epoch_list, train_MAE_list]], ['train_MAE-epoch'], ['#03658C'], 'epoch', 'train_MAE', f'Training_Recording/{jobtype}/{TIME}/Plot/train_MAE-epoch_{TIME}.png')
    if not KC:
        loss_epoch([[epoch_list, val_MAE_list]], ['val_MAE-epoch'], ['#03658C'], 'epoch', 'val_MAE', f'Training_Recording/{jobtype}/{TIME}/Plot/val_MAE-epoch_{TIME}.png')
    loss_epoch([[epoch_list, test_MAE_list]], ['test_MAE-epoch'], ['#03658C'], 'epoch', 'test_MAE', f'Training_Recording/{jobtype}/{TIME}/Plot/test_MAE-epoch_{TIME}.png')


if __name__ == '__main__':
    with open('model_parameters.yml') as mp:
        param = yaml.full_load(mp)
    mode = param['mode']
    print(mode)
    
    if 'K' in mode:
        if param['scheduler'] != 'Exp':
            change_yaml_KC('model_parameters.yml','scheduler','Exp')
        if param['split_method'] != 'manual':
            change_yaml_KC('model_parameters.yml','split_method','manual')
    else:
        if param['scheduler'] != 'plateau':
            change_yaml_KC('model_parameters.yml','scheduler','plateau')

    if mode =='Sg':
        main()
    if mode =='Sg_mK':
        main()
    elif mode == 'Cir':
        for cir_num in range(30):
            change_yaml_KC('model_parameters.yml','seed',cir_num)
    elif mode.startswith('KC'):
        if mode == 'KC':
            npy_path = param['KC_path']
            for i in range(5):
                path_rp = f'{npy_path}split_{i}.npy'
                change_yaml_KC('model_parameters.yml','SPLIT_file',path_rp)
                print(f'npy--{path_rp}') 
                main()    
        if mode == 'KC_R':
            for random_seed in [2,3,5,6,7,11,13,15,17,19,24,33,60]:
                npy_path = KF_random(random_seed+59)
                for i in range(5):
                    path_rp = f'{npy_path}split_{i}.npy'
                    change_yaml_KC('model_parameters.yml','SPLIT_file',path_rp)
                    print(f'npy--{path_rp}') 
                    main()
    
    elif mode.startswith('Sc'):
        Sc_line = param['Sc_line']
        Sc_out = param['Sc_out']
        if Sc_line == None or Sc_out == []:
            print('!!!______erorr_no_Sc_data______!!!')
        else:
            if mode == 'Sc_p':
                for Sc_num in Sc_out:
                    change_yaml_KC('model_parameters.yml',Sc_line,Sc_num)
                    print(f'{Sc_line} changed into {Sc_num}')
                    main() 
            if mode == 'Sc_K':
                for Sc_num in Sc_out:
                    change_yaml_KC('model_parameters.yml',Sc_line,Sc_num)
                    print(f'{Sc_line} changed into {Sc_num}')
                    npy_path = param['KC_path']
                    if not npy_path.endswith('/'):
                        npy_path = npy_path+'/'
                    for i in range(5):
                        path_rp = f'{npy_path}split_{i}.npy'
                        change_yaml_KC('model_parameters.yml','SPLIT_file',path_rp)
                        print(f'npy--{path_rp}') 
                        main() 
            if mode == 'Sc_KG':
                if isinstance(Sc_line,list) and isinstance(Sc_out,list):
                    graph_hp_lists = graph_hyperparameter(Sc_line,Sc_out)
                    if isinstance(graph_hp_lists,list):
                        for graph_hp_list in graph_hp_lists:
                            for graph_hp_num,graph_hp in enumerate(graph_hp_list):
                                change_yaml_KC('model_parameters.yml',Sc_line[graph_hp_num],graph_hp)
                            print(f'G_hp are {graph_hp_list}')
                            npy_path = param['KC_path']
                            if not npy_path.endswith('/'):
                                npy_path = npy_path+'/'
                            for i in range(5):
                                path_rp = f'{npy_path}split_{i}.npy'
                                change_yaml_KC('model_parameters.yml','SPLIT_file',path_rp)
                                print(f'npy--{path_rp}') 
                                main()           
                else:
                      print('!!!______error_GraphHyperparameter______!!!')
    else:
        print('!!!______error_mode______!!!')