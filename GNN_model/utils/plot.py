import matplotlib.pyplot as plt
import torch
from typing import List

def pred_tgt(x,
             y,
             MAE: float,
             R2: float,
             output_path: str,
             dot_color: str = 'b',
             line_color: str = 'r',
             xlabel: str = 'Target',
             ylabel: str = 'Predict'
             ):
    '''scatter plot
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    lower_limit = min([min(x), min(y)])
    upper_limit = max([max(x), max(y)])
    displacement = (upper_limit - lower_limit) * 0.1
    plt.figure(
        figsize = (10, 10),
        dpi = 300
        )
    plt.scatter(
        x,
        y,
        10,
        color = dot_color
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(
        [lower_limit - displacement, upper_limit + displacement],
        [lower_limit - displacement, upper_limit + displacement],
        color = line_color,
        linewidth = 0.5,
        label = 'y = x'
        )
    plt.text(
        lower_limit,
        upper_limit,
        f'MAE = {MAE:.4f}\nR2 = {R2:.4f}'
        )
    plt.legend(loc = 'best')
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()

def loss_epoch(
    data_list: List[List[List[float]]],
    label_list: List[str],
    color_list: List[str],
    xlabel: str,
    ylabel: str,
    output_path: str,
    fontsize: float = 20,
    ):
    '''line graph
    '''
    assert len(data_list) == len(label_list), 'Data_list and Label_list are not of the same length.'
    assert len(data_list) <= len(color_list), 'There are not enough colors.'
    plt.figure(
        figsize = (10, 10),
        dpi = 300
        )
    for i, data in enumerate(data_list):
        x = data[0]
        y = data[1]
        plt.plot(x, y, color = color_list[i], label = label_list[i])
    plt.legend(loc = 'best', fontsize = fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(
        output_path,
        bbox_inches = 'tight',
        dpi = 300
        )
    plt.close()

