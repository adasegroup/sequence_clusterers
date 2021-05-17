import os
import pandas as pd
import torch
import numpy as np
import pickle

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def compare(a,b):
    tmp1 = int(a[:-4])
    tmp2 = int(b[:-4])
    return tmp1 - tmp2

def get_partition(df, num_of_steps, num_of_classes, end_time = None):
    if end_time == None:
        end_time = df['time'][len(df['time'])-1]
    res = torch.zeros(num_of_steps, num_of_classes + 1)
    dt = end_time/num_of_steps
    res[:,0] = end_time/num_of_steps
    for i in range(len(df['time'])):
        k = int(df['time'][i]/dt)
        if k == num_of_steps:
            k-=1
        res[k,int(df['event'][i])+1] += 1
    return res

def get_dataset(path_to_files, n_classes, n_steps):
    files = os.listdir(path_to_files)
    target = None
    if 'clusters.csv' in files:
        files.remove('clusters.csv')
        target = torch.Tensor(pd.read_csv(path_to_files+'/clusters.csv')['cluster_id'])
    #print(target)
    files = sorted(files, key = cmp_to_key(compare))
    data = torch.zeros(len(files), n_steps, n_classes + 1)
    for i, f in enumerate(files):
        df = pd.read_csv(path_to_files+'/'+f)
        data[i,:,:] = get_partition(df, n_steps, n_classes)
    return data, target

path_to_files = 'data/simulated_Hawkes/K4_C5'
N_STEPS = 128
N_CLUSTERS = 4
EPS = 0.8
n_runs = 5

data, target = get_dataset(path_to_files, 5, N_STEPS)

#with open('utils/data.pkl', 'rb') as f:
#    data = pickle.load(f)
#    
#with open('utils/clusters.pkl', 'rb') as f:
#    target = pickle.load(f)
indices = np.random.permutation(data.shape[0])
data_train, target_train = data[indices[:9*data.shape[0]//10]],\
                           target[indices[:9*data.shape[0]//10]]
data_test, target_test = data[indices[9*data.shape[0]//10:]],\
                           target[indices[9*data.shape[0]//10:]]

import torch
from models.models import LSTM_cluster_point_processes
from utils.trainers import Trainer_clusterwise

device = 'cuda:1'

all_purs = []
all_purs_val = []
i = 0
while i<n_runs:
#test_param = [1.0001,1.0002,1.0005,1.001,1.002,1.003,1.005,1.01,1.02,1.05,1.1]
#success = []
#for i in test_param:
    print('Run {}/{}'.format(i+1, n_runs))
    #print('alpha =', i)
    model = LSTM_cluster_point_processes(6,128, 3, 5, N_CLUSTERS, N_STEPS, dropout = 0.3).to(device)
    #model.load_state_dict(torch.load('reserv.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr =0.1, weight_decay = 1e-5)
    trainer = Trainer_clusterwise(model, optimizer, device, data, data_test,\
                              target, target_test, N_CLUSTERS, alpha = 1.0003, beta = 0.001,\
                              epsilon = 1e-8, l = 30, eps = 0.0/N_CLUSTERS,\
                              max_epochs = 25, max_m_step_epochs = 50,\
                              lr_update_tol = 25, lr_update_param = 0.95,\
                             batch_size = 400)

    losses, purs, purs_val, cluster_part = trainer.train()
    if cluster_part == None:
        continue
    if cluster_part < EPS/N_CLUSTERS:
        print("Degenerate solution")
        continue
    if losses:
        all_purs+=purs
        all_purs_val += purs_val
        i+=1

import pickle

with open('K4all_purs_10003.pkl', 'wb') as f:
    pickle.dump(all_purs, f)
with open('K4all_purs_val_10003.pkl', 'wb') as f:
    pickle.dump(all_purs_val, f)
with open('all_purs_copy.pkl', 'wb') as f:
    pickle.dump(all_purs, f)
with open('all_purs_val_copy.pkl', 'wb') as f:
    pickle.dump(all_purs_val, f)