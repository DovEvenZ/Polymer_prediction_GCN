from itertools import product

def graph_hyperparameter(A, B):
    # 使用 itertools.product 计算笛卡尔积，*B 是解包操作符，用来传递多个参数给 product 函数。
    if len(A) == len(B):
        combinations = list(product(*B))
        
        # 将组合转换为列表形式返回
        all_possible_C = [list(combination) for combination in combinations]
    else:
        all_possible_C = 'error'

    return all_possible_C