import torch
from typing import Dict
from Net.FNN import FNN_1, FNN_2

net_dict = {
    1: FNN_1,
    2: FNN_2
}

def gen_model(param: Dict, dataset, ) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    restart_model = param['restart_model']
    hidden_layer_info = param['hidden_layer']
    num_input = len(param['feature_list'])
    num_output = len(param['target_list'])
    net = net_dict[len(hidden_layer_info)](num_input, *hidden_layer_info, num_output)
    model = net.to(device)
    return model

def gen_optimizer(param: Dict, model):
    if param['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = param['lr'])
    elif param['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr = param['lr'])
    elif param['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = param['lr'])
    else:
        raise NotImplementedError("Optimizer not implemented.")
    return optimizer

def gen_scheduler(param: Dict, optimizer):
    if param['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode = 'min',
            factor = param['scheduler_factor'], patience = param['scheduler_patience'],
            min_lr = param['scheduler_min_lr']
            )
    else:
        print('No scheduler')
        scheduler = None
    return scheduler
