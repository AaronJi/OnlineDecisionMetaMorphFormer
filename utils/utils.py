import numpy as np
import torch

def get_nparray_from_str(s):
    v = s.split(',')
    return np.array([float(vv) for vv in v])

def get_str_from_1darray(v):
    round_digits = 4
    sv = [str(round(vv, round_digits)) for vv in v]
    return ','.join(sv)

def reform_window_len(data, window_len, index_end=None, padding_num=None, batch_dim=1, data_form='np', device=None):
    assert index_end is None or index_end >= 0

    if index_end is None:
        window_data = data[-window_len:]
    else:
        index_start = max(index_end - window_len + 1, 0)
        window_data = data[index_start:index_end+1]

    # padding
    if padding_num is not None:
        window_data = padding_to_window(window_data, window_len, padding_num=padding_num, batch_dim=batch_dim, data_form=data_form, device=device)
    return window_data

def padding_to_window(win_data, window_len, padding_num=None, batch_dim=1, data_form='np', device=None):

    if padding_num is None:
        padding_num = 0

    if batch_dim == 2:
        tlen = win_data.shape[0]
        dim = win_data.shape[2]
        dummy_array = np.ones((window_len - tlen, win_data.shape[1], dim))
        cat_dim = 0
    elif batch_dim == 1:
        tlen = win_data.shape[0]
        dim = win_data.shape[1]
        dummy_array = np.ones((window_len - tlen, dim))
        cat_dim = 0
    else:
        tlen = win_data.shape[0]
        dummy_array = np.ones(window_len - tlen)
        cat_dim = 0

    if data_form == 'np':
        win_data = np.concatenate([padding_num * dummy_array, win_data],  axis=cat_dim)
    else:
        #print('&*' * 20)
        #print(win_data)
        #print(torch.tensor(dummy_array).double())
        padding_tensor = padding_num * torch.tensor(dummy_array, device=device).to(torch.float32)
        win_data = torch.cat([padding_tensor, win_data], dim=cat_dim)
        #print(win_data)
        #exit(5)
    return win_data

def get_last_from_seq(seq):
    if seq is None:
        return seq

    if isinstance(seq, dict):
        last = {}
        for ot in seq:
            last[ot] = seq[ot][:, -1, :]
    else:
        last = seq[:, -1, :]
    return last

def get_dummy_seq_from_var(var, axis_time=1):
    if var is None:
        return var

    if isinstance(var, dict):
        seq = {}
        for ot in var:
            seq[ot] = var[ot].unsqueeze(axis_time)
    else:
        seq = var.unsqueeze(axis_time)
    return seq