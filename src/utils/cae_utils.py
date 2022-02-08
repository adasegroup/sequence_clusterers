import  numpy as np
import torch


def dict_to_pk(dict):
    pk = []
    for ss in dict.values():
        pk.extend(ss)
    return pk


def arr_func(events, T_j, delta_T, fws_func):
    events_fws = dict()
    for p_k in events:

        fws_val = fws_func(p_k, delta_T)
        if type(p_k) == dict:
            p_k = dict_to_pk(p_k)
        p_k1 = tuple(p_k)
        if p_k1 not in events_fws.keys():
            events_fws[p_k1] = []
            events_fws[p_k1].append(fws_val)
        else:
            events_fws[p_k1].append(fws_val)

    array = []
    for val in events_fws.values():
        array.append(list(val[0]))
    return array, events_fws


def fws(p, t1, t2):
    n = sum(list(map(int, (p >= t1) & (p <= t2))))
    return min(int(np.log2(n + 1)), 9)


def multiclass_fws_array(user_dict, time_partition):
    fws = []
    for event, subseq in user_dict.items():
        arr = fws_numerical_array(subseq, time_partition)
        fws.append(arr)
    return fws


def fws_numerical_array(p, array):
    fws_array = []
    for i in range(1, len(array)):
        fws_array.append(fws(p, array[i - 1], array[i]))
    return fws_array


def events_tensor(events_fws):
    keys_list = list(events_fws.keys())
    full_tensor_batch = torch.tensor([], dtype=torch.float32)
    for key in keys_list:

        ten = torch.tensor(events_fws[key]).unsqueeze(0)
        if ten.shape[1] == 1:
            full_tensor_batch = torch.cat((full_tensor_batch.float(), ten.float()), dim=0)
        else:
            for i in range(ten.shape[1]):
                ten2 = ten[:, i, :].unsqueeze(0)
                full_tensor_batch = torch.cat((full_tensor_batch.float(), ten2.float()), dim=0)
    if len(full_tensor_batch.shape) == 4:
        full_tensor_batch = full_tensor_batch.squeeze(axis=1)
    return full_tensor_batch

