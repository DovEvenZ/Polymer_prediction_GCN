from typing import Dict
import json

class read_log():
    def __init__(self, log_file: str) -> None:
        self.log_file = log_file

    def get_performance(self) -> Dict:
        epoch_list = []
        lr_list = []
        loss_list = []
        train_MAE_list = []
        train_R2_list = []
        train_ARD_list = []
        val_MAE_list = []
        val_R2_list = []
        val_ARD_list = []
        test_MAE_list = []
        test_R2_list = []
        test_ARD_list = []

        log_info_dict = {
            'Epoch': epoch_list,
            'LR': lr_list,
            'Loss': loss_list,
            'Train MAE': train_MAE_list,
            'Train R2': train_R2_list,
            'Train ARD': train_ARD_list,
            'Val MAE': val_MAE_list,
            'Val R2': val_R2_list,
            'Val ARD': val_ARD_list,
            'Test MAE': test_MAE_list,
            'Test R2': test_R2_list,
            'Test ARD': test_ARD_list
        }

        with open(self.log_file) as lf:
            text = lf.readlines()
        for line in text:
            if line.startswith('Epoch'):
                info_list = line.split(',')
                for info in info_list:
                    for key in log_info_dict.keys():
                        if key in info:
                            log_info_dict[key].append(float(info.split()[-1]))
            if line.startswith('===='):
                break

        return log_info_dict

    def get_feature(self) -> Dict:
        with open(self.log_file) as lf:
            text = lf.readlines()
        param = json.loads(text[1])
        feature = {
            'node': param['node_attr_list'],
            'edge': param['edge_attr_list'],
            'graph': param['graph_attr_list']
            }
        return feature

    def restart(self, start_epoch: int):
        with open(self.log_file) as lf:
            text = lf.readlines()
        pre_log_text = []
        i = 1
        for line in text:
            if i == start_epoch:
                break
            if line.startswith('Epoch'):
                pre_log_text.append(line)
                i += 1
        return pre_log_text
