import torch
import tarfile
import pickle
import pandas
import json
import argparse
from pathlib import Path
import numpy as np
import shutil
from shutil import copyfile
import os
import re
import pandas as pd
import sys
from numpy import asarray
from numpy import savetxt
sys.path.append("..")
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--maxlen', type=int, default=500, help='maximum length of sequence')
    parser.add_argument('--ext', type=str, default='tar.gz', help='extention of files with sequences')
    parser.add_argument('--datetime', type=bool, default=False, help='if time values in event sequences are represented in datetime format')
    parser.add_argument('--save_dir', type=str, default = './', help='path to save results')
    parser.add_argument('--maxsize', type=int, default=None, help='max number of sequences')
    args = parser.parse_args()
    return args
def tranform_data(args):
    """
    Loads the sequences saved in the given directory.
    Args:
        data_dir    (str, Path) - directory containing sequences
        save_dir - directory for saving transform data
        maxsize     (int)       - maximum number of sequences to load
        maxlen      (int)       - maximum length of sequence, the sequences longer than maxlen will be truncated
        ext         (str)       - extension of files in data_dir directory
        datetime    (bool)      - variable meaning if time values in files are represented in datetime format
             
    """
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir)
    maxsize = args.maxsize
    maxlen = args.maxlen  
    ext = args.ext
    datetime = args.datetime
    classes = set()
    nb_files = 0
    time_col = 'time'
    event_col = 'event'
    if ext == "tar.gz" or "pkl":
        if ext == "pkl":
            classes = set()
            print('hi(')
            df = pd.read_pickle(Path(data_dir, "fx_data.pkl"))[:maxsize]
            for i in range (df.shape[0]):
                data = {'time': [df.iloc[i]['time']], 'event': [df.iloc[i]['ud']]}
                df_data = pd.DataFrame(data=data)
                classes = classes.union(set(df_data[event_col].unique()))
                os.mknod(Path(save_dir,f'{i+1}.csv'))
                df_data.to_csv(Path(save_dir,f'{i+1}.csv'))
                print (f'Reading {i} out of {df.shape[0]}\n')
            seq_nmb = df.shape[0]
            classes = list(classes)
    
            info = {
                    "classes": list(map(int,classes)),
                    "seq_nmb": int(seq_nmb)
                }
        if ext == "tar.gz":
            with tarfile.open(data_dir, "r:gz") as tar:
                classes = set()
                fp = tar.extractfile("synthetic_hawkes_data")
                df = pickle.load(fp)
                fp.close()
            for i in range (maxsize):
                data = {'time': [df[3][i]], 'event': [df[4][i]]}
                df_data = pd.DataFrame(data=data)
                classes = classes.union(set(df_data[event_col].unique()))
                os.mknod(Path(save_dir,f'{i+1}.csv'))
                df_data.to_csv(Path(save_dir,f'{i+1}.csv'))
                print (f'Reading {i} \n')
            seq_nmb = maxsize
            classes = list(classes)
            info = {
                    "classes": list(map(int,classes)),
                    "seq_nmb": int(seq_nmb)
                }
    if ext == "csv" or ext =="txt":
        classes = set()
        for file in sorted(os.listdir(data_dir), key=lambda x: int(re.sub(fr'.{ext}', '', x)) if re.sub(fr'.{ext}', '', x).isdigit() else 0):
            if file.endswith(f'.{ext}') and re.sub(fr'.{ext}', '', file).isnumeric():
                if maxsize is None or nb_files < maxsize:
                    nb_files += 1
                else:
                    break
                df = pd.read_csv(Path(data_dir, file))
                classes = classes.union(set(df[event_col].unique()))
                if datetime:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df[time_col] = (df[time_col] - df[time_col][0]) / np.timedelta64(1,'D')
                #if maxlen > 0:
                    #df = df.iloc[:maxlen]
                df.to_csv(Path(save_dir,file.replace(ext,'csv')))
        seq_nmb = nb_files
        classes = list(classes)

        info = {
                "classes": classes,
                "seq_nmb": seq_nmb
            }
    gt_ids = None
    if args.ext == "pkl":
        with open(Path(args.data_dir, "fx_labels"), "rb") as fp:
            gt_ids = pickle.load(fp)[:maxsize]
            labels = np.unique(gt_ids)
            gt_data = []
            for i in range (len(gt_ids)):
                gt_data.append(labels[np.nonzero(gt_ids[i] == labels)])
            gt = {'cluster_id': gt_data}
            print(gt_data)
            gt_table = pd.DataFrame(data=gt)
            gt_table.to_csv(Path(save_dir, 'clusters.csv'))
    if Path(args.data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(args.data_dir, 'clusters.csv'))[:(maxsize)]
        gt_ids.to_csv(Path(save_dir, 'clusters.csv'))
    
    print(info)
    with open(Path(args.save_dir, 'info.json'), "w") as f:
            json.dump(info, f, indent=4)
    return 'Data transforming completed' 


args = parse_arguments()
print(args)
tranform_data(args)
