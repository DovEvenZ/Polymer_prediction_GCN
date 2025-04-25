import os, shutil, json
from torch_geometric.loader import DataLoader
from typing import Dict

from utils.time import convert_time

class file_processing():
    def __init__(self, jobtype: str, TIME: str, param: Dict) -> None:
        self.jobtype = jobtype
        self.TIME = TIME
        self.param = param
        self.log_file = f'Training_Recording/{self.jobtype}/{self.TIME}/training_{self.TIME}.log'

    def pre_make(self) -> None:
        os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}')
        shutil.copy('model_parameters.yml', f'Training_Recording/{self.jobtype}/{self.TIME}')
        os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}/Plot')
        os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}/Model')
        os.makedirs(f'Training_Recording/{self.jobtype}/{self.TIME}/Model/checkpoint')

    def basic_info_log(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        with open(self.log_file, 'a') as tl:
            data_abspath = os.path.abspath(self.param['path'])
            tl.write(f'data_path: {data_abspath}\n')
            tl.write(f'{json.dumps(self.param)}\n')
            tl.write(f'size of test set: {len(test_loader.dataset)}\n')
            tl.write(f'size of val set: {len(val_loader.dataset)}\n')
            tl.write(f'size of training set: {len(train_loader.dataset)}\n')
            tl.write("Begin training...\n")

    def ending_log(
        self,
        best_MAE_val: float,
        best_epoch_MAE: int,
        best_epoch_MAE_info: str,
        best_ARD_val: float,
        best_epoch_ARD: int,
        best_epoch_ARD_info: str,
        end_time: float,
        start_time: float,
        ) -> None:
        tot_time = end_time - start_time
        epoch_time = tot_time / self.param['epoch_num']
        with open(self.log_file, 'a') as tl:
            tl.write('\n===================================================================\n')
            tl.write(f'Best_MAE_val: {best_MAE_val}\n')
            tl.write(f'Best_epoch_MAE: {best_epoch_MAE}\n')
            tl.write(f'{best_epoch_MAE_info}\n')
            tl.write(f'Best_ARD_val: {best_ARD_val}\n')
            tl.write(f'Best_epoch_ARD: {best_epoch_ARD}\n')
            tl.write(f'{best_epoch_ARD_info}\n')
            hours, minutes, seconds = convert_time(tot_time)
            tl.write(f'Total_time: {hours} h {minutes} m {seconds} s\n')
            hours, minutes, seconds = convert_time(epoch_time)
            tl.write(f'Time_per_epoch: {hours} h {minutes} m {seconds} s\n')
