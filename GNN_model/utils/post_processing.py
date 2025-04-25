from typing import Dict
import json

class read_log():
    def __init__(self, log_file: str) -> None:
        self.log_file = log_file

    def get_performance(self) -> Dict:
        epoch_list = []
        loss_list = []
        train_MAE_list = []
        val_MAE_list = []
        test_MAE_list = []

        with open(self.log_file) as lf:
            text = lf.readlines()
        for line in text:
            if line.startswith('Epoch'):
                epoch = int(line.split(',')[0].split()[-1])
                epoch_list.append(epoch)
                loss = float(line.split(',')[2].split()[-1])
                loss_list.append(loss)
                train_MAE = float(line.split(',')[3].split()[-1])
                train_MAE_list.append(train_MAE)
                val_MAE = float(line.split(',')[5].split()[-1])
                val_MAE_list.append(val_MAE)
                test_MAE = float(line.split(',')[7].split()[-1])
                test_MAE_list.append(test_MAE)
        log_info_dict = {
            'epoch': epoch_list,
            'loss': loss_list,
            'train_MAE': train_MAE_list,
            'val_MAE': val_MAE_list,
            'test_MAE': test_MAE_list
        }

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
