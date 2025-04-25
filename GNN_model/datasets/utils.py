'''utils for gnn attribute.
'''

def one_hot(hot_idx: int, total_len: int):
    '''generate one hot repr according to selected index and
    total length.
    
    Args:
        hot_idx: the index chosen to be 1.
        total_len: how long should the repr be.
        
    Return:
        one_hot_list: [0, 0, 1, 0] for hot_idx=2, total_len=4.
    '''
    one_hot_list = []
    for i in range(total_len):
        if i == hot_idx:
            one_hot_list.append(1)
        else:
            one_hot_list.append(0)
    
    return one_hot_list
