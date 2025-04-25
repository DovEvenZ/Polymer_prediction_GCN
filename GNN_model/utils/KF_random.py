import os
import yaml
import time
import numpy as np
from sklearn.model_selection import KFold

def KF_random(num):
    with open('model_parameters.yml') as mp:
        param = yaml.full_load(mp)
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    jobtype = param['jobtype']
    # 获取数据集长度
    num_samples = param['num_samples']

    # 初始化 KFold 分割器
    kf = KFold(n_splits=5, shuffle=True, random_state=num)

    kf_dic = {}
    # 遍历每个折叠
    out_path = f'npy4CV/{jobtype}/{TIME}/'
    os.makedirs(out_path)
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(num_samples))):
        name_kf = f"Fold {fold+1}"
        kf_dic[name_kf] = test_idx.tolist()
        t2_set = [train_idx.tolist(),[],test_idx.tolist()]
        split_array = np.array([np.array(i, dtype=np.int64) for i in t2_set], dtype='object')
        np.save(f'{out_path}split_{fold}.npy', split_array)

    return out_path