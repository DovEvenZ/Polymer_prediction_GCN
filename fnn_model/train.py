import torch
import math
import yaml
import time
import os
import shutil
from functools import partial
from typing import Dict, List

from utils.setup_seed import setup_seed
from utils.plot import loss_epoch
from utils.plot_model import scatterFromModel
from utils.post_processing import read_log
from utils.calc_error import calc_error
from utils.data_processing import data_processing
from utils.gen_model import gen_model, gen_optimizer, gen_scheduler
from utils.train import train
from utils.file_processing import file_processing
from utils.attr_filter import attr_filter

from utils.change_yaml_KC import change_yaml_KC

def main(param: Dict):
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    jobtype = param['jobtype']
    fp = file_processing(jobtype, TIME, param)
    fp.pre_make()
    log_file = fp.log_file

    seed = param['seed']
    setup_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    DATA = data_processing(param, reprocess = True)
    dataset = DATA.dataset
    mean, std = DATA.mean, DATA.std
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader

    fp.basic_info_log(train_loader, val_loader, test_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gen_model(param, dataset)
    optimizer = gen_optimizer(param, model)
    scheduler = gen_scheduler(param, optimizer)

    epoch_num = param['epoch_num']
    best_MAE_val = math.inf
    best_ARD_val = math.inf
    best_epoch_MAE = None
    best_epoch_MAE_info = None
    best_epoch_ARD = None
    best_epoch_ARD_info = None

    restart_model = param['restart_model']
    if restart_model:
        state_dict = torch.load(restart_model)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        pre_dir = os.path.dirname(os.path.dirname(os.path.dirname(restart_model)))
        pre_TIME = os.path.basename(pre_dir)
        pre_log_file = os.path.join(pre_dir, f'training_{pre_TIME}.log')
        shutil.copy(pre_log_file, f'Training_Recording/{jobtype}/{TIME}/pre.log')
        pre_log_text = read_log(pre_log_file).restart(start_epoch)
        with open(log_file, 'a') as tl:
            tl.writelines(pre_log_text)
        pre_log_info = read_log(pre_log_file)
        pre_epoch_list: List = pre_log_info.get_performance()['Epoch'][:start_epoch - 1]
        pre_val_MAE_list: List = pre_log_info.get_performance()['Val MAE'][:start_epoch - 1]
        pre_val_ARD_list: List = pre_log_info.get_performance()['Val ARD'][:start_epoch - 1]
        best_MAE_val = min(pre_val_MAE_list)
        best_ARD_val = min(pre_val_ARD_list)
        best_epoch_MAE = pre_epoch_list[pre_val_MAE_list.index(best_MAE_val)]
        best_epoch_ARD = pre_epoch_list[pre_val_ARD_list.index(best_ARD_val)]
    else:
        start_epoch = 1
        best_MAE_val = math.inf
        best_ARD_val = math.inf
        best_epoch_MAE = None
        best_epoch_MAE_info = None
        best_epoch_ARD = None
        best_epoch_ARD_info = None

    criteria_list = param['criteria_list']

    error_class = partial(calc_error, device = device, mean = mean[-1], std = std[-1], transform = param['target_transform'])
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epoch_num+1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model, train_loader, optimizer, torch.nn.MSELoss(), device)
        train_error = error_class(train_loader, model)
        val_error = error_class(val_loader, model)
        test_error = error_class(test_loader, model)
        MAE_train, R2_train, ARD_train = train_error.MAE(), train_error.R2(), train_error.ARD()
        MAE_val, R2_val, ARD_val = val_error.MAE(), val_error.R2(), val_error.ARD()
        MAE_test, R2_test, ARD_test = test_error.MAE(), test_error.R2(), test_error.ARD()
        scheduler.step(MAE_val)

        info = (f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                f'Train MAE: {MAE_train:.7f}, Train R2: {R2_train:.7f}, Train ARD: {ARD_train:.7f}, '
                f'Val MAE: {MAE_val:.7f}, Val R2: {R2_val:.7f}, Val ARD: {ARD_val:.7f}, '
                f'Test MAE: {MAE_test:.7f}, Test R2: {R2_test:.7f}, Test ARD: {ARD_test:.7f}\n')
        if 'MAE' in criteria_list:
            if MAE_val <= best_MAE_val:
                model.eval()
                best_MAE_val = MAE_val
                best_epoch_MAE = epoch
                best_epoch_MAE_info = info
                state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state_dict, f'Training_Recording/{jobtype}/{TIME}/Model/best_model_MAE_{TIME}.pth')
        if 'ARD' in criteria_list:
            if ARD_val <= best_ARD_val:
                model.eval()
                best_ARD_val = ARD_val
                best_epoch_ARD = epoch
                best_epoch_ARD_info = info
                state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state_dict, f'Training_Recording/{jobtype}/{TIME}/Model/best_model_ARD_{TIME}.pth')

        if epoch % param['output_step'] == 0:
            with open(log_file, 'a') as tl:
                tl.write(info)
        if epoch % param['model_save_step'] == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, f'Training_Recording/{jobtype}/{TIME}/Model/checkpoint/ckpt_{TIME}_{epoch:0{len(str(epoch_num))}d}.pth')
    end_time = time.perf_counter()

    fp.ending_log(best_MAE_val, best_epoch_MAE, best_epoch_MAE_info, best_ARD_val, best_epoch_ARD, best_epoch_ARD_info, end_time, start_time)

    for criteria in criteria_list:
        scatterFromModel(
            f'Training_Recording/{jobtype}/{TIME}/Model/best_model_{criteria}_{TIME}.pth',
            param,
            DATA,
            f'Training_Recording/{jobtype}/{TIME}/Plot/'
            )

    plotting_objects = param['plotting_objects']
    log_info_dict = read_log(log_file).get_performance()
    epoch_list = log_info_dict['Epoch']
    for key, value in log_info_dict.items():
        if not key in plotting_objects:
            continue
        loss_epoch(
            [[epoch_list, log_info_dict[key]]],
            [f'{key}-Epoch'],
            ['#03658C'],
            'Epoch',
            f'{key}',
            f'Training_Recording/{jobtype}/{TIME}/Plot/{key}-Epoch_{TIME}.png'
            )


if __name__ == '__main__':
    with open('model_parameters.yml') as mp:
        param: Dict = yaml.full_load(mp)

    if param['feature_filter_mode'] == 'one_by_one':
        attr_filter(main, param)
    elif param['feature_filter_mode'] == 'file':
        with open('feature_filter.yml') as ff:
            feature_dict: Dict = yaml.full_load(ff)
        for idx, feature in feature_dict.items():
            param['node_attr_list'] = feature['node_attr_list']
            param['edge_attr_list'] = feature['edge_attr_list']
            param['graph_attr_list'] = feature['graph_attr_list']
            main(param)
    elif not param['feature_filter_mode']:
        main(param)
