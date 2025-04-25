import torch
import os
from typing import Dict
from functools import partial
from utils.gen_model import gen_model
from utils.data_processing import data_processing
from utils.calc_error import calc_error
from utils.plot import scatter

def scatterFromModel(model_path: str, param: Dict, DATA: data_processing, output_dir: str):
    train_loader = DATA.train_loader
    val_loader = DATA.val_loader
    test_loader = DATA.test_loader

    model = gen_model(param, DATA.dataset)
    _model = torch.load(model_path, map_location = 'cuda' if torch.cuda.is_available() else 'cpu')
    if model_path.endswith('pkl'):
        model = _model
    elif model_path.endswith('pth'):
        model.load_state_dict(_model['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    error_class = partial(
        calc_error,
        model = model,
        device = device,
        mean = DATA.mean[-1],
        std = DATA.std[-1],
        transform = param['target_transform']
        )
    train_error = error_class(train_loader)
    val_error = error_class(val_loader)
    test_error = error_class(test_loader)
    error_dict = {
        'train': train_error,
        'val': val_error,
        'test': test_error
        }

    file_name = os.path.splitext(os.path.basename(model_path))[0]
    for key, value in error_dict.items():
        scatter(
            [value.target, value.pred],
            scatter_label = key,
            output_path = os.path.join(output_dir, f'{file_name}_{key}.png'),
            )
    scatter(
        [train_error.target, train_error.pred],
        [test_error.target, test_error.pred],
        scatter_label = ['train', 'test'],
        output_path = os.path.join(output_dir, f'{file_name}_train_test.png'),
        )
