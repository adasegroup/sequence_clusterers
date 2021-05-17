#
#This file contains implementation of functions needed to perform Cohortney algorithm
#
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange


def dict_to_pk(dict):
  pk = []
  for ss in dict.values():
    pk.extend(ss)
  return pk


#transforming data to the array taking into account an event type
def sep_hawkes_proc(user_list, event_type):
  sep_seqs = []
  for user_dict in user_list:
    sep_seqs.append(np.array(user_dict[event_type], dtype = np.float32))

  return sep_seqs


#transforming data to the tensor without putting attention at event type
def hawkes_process_wo_event_types(ss):
  sep_seqs = torch.tensor([],dtype=torch.float32)
  for i in range(len(ss)):

      sep_seqs = torch.cat((sep_seqs, torch.unsqueeze(ss[i].T[0],0)), 0)
  return sep_seqs


def fws(p, t1, t2):
  n = sum(list(map(int, (p >= t1) & (p <= t2))))
  return min(int(np.log2(n+1)), 9)


def multiclass_fws_array(user_dict, time_partition):
  fws = []
  for event, subseq in user_dict.items():
    arr = fws_numerical_array(subseq, time_partition)
    fws.append(arr)
  return fws


def fws_numerical_array(p, array):
  fws_array = []
  for i in range(1, len(array)):
    # if type(p) == dict:
    #       p = dict_to_pk(p)
    fws_array.append(fws(p, array[i-1], array[i]))
  # fws_array = tuple(fws_array)
  return fws_array


def computing_cohortney(grid, events, fws_func, n):
  events_clusters = dict() #initial clusters dict. key - cluster defining triplet, value - list of sequences
  events_triplets = dict() # key -sequence, value -list of clusters

  for i in range(len(grid)):
    for k in range(n):
      T_j = grid[i]
      Delta_T = np.linspace(0, grid[i], 2**k)
      Delta_T = Delta_T[Delta_T< int(T_j)]
      Delta_T = tuple(Delta_T)
      for p_k in events:
        hs = ""
        fws_val =  fws_func(p_k, Delta_T)
        fws_val = hs.join([str(el) for el in  list(np.array(fws_val).reshape(1,-1).squeeze())])
       
        triplet = (int(T_j), Delta_T, fws_val)
        if type(p_k) == dict:
          p_k = dict_to_pk(p_k)
        if triplet not in events_clusters.keys():
          events_clusters[triplet] = []
          events_clusters[triplet].append(p_k) 
        else:
          events_clusters[triplet].append(p_k) 
        p_k1 = tuple(p_k)
        if p_k1 not in events_triplets.keys():
          events_triplets[p_k1] = []
          events_triplets[p_k1].append(triplet)
        else:
          events_triplets[p_k1].append(triplet)

  return events_clusters, events_triplets

#dropping clusters with less then N sequences in it
def optimalClusters(events_clusters, N):
  optimal_clusters = {}
  for k,v in events_clusters.items():
    if len(v)>= N:
      optimal_clusters[k] = v
  return optimal_clusters

def new_triplets_for_seq(boarder_time, new_seq, fws_func, n, grid):
  t = boarder_time
  p_k = new_seq
  grid_cut = grid[grid<t]
  p_new_triplets = []
  n = n
  for k in range(n):
      Delta_T = np.linspace(0, grid_cut[-1], 2**k)
      Delta_T = tuple(Delta_T)
      hs = ""
      fws_val =  fws_func(p_k, Delta_T)
      fws_val = hs.join([str(el) for el in list(np.array(fws_val).reshape(1,-1).squeeze())])
       
      triplet = (int(t), Delta_T, fws_val)

      if triplet not in p_new_triplets:
        p_new_triplets.append(triplet)
  return p_new_triplets

def diffLit(s,t):
    return int(s != t)


def EditDIstTD(i,j):
    sub,deel, ins = 0,0,0
    el =0
    if d[i][j] == 2147483646:
        if i == 0:
            d[i][j] = j
        elif j == 0:
            d[i][j]= i
        else:
            ins = EditDIstTD(i, j-1)+1
            deel = EditDIstTD(i-1, j)+1
            sub = EditDIstTD(i-1, j-1) + diffLit(A[j-1], B[i-1])
            d[i][j] = min(ins, deel, sub)

    return d[i][j]


def Levenshtein(A, B):
  for i in range(1,len(B)+1):
      for j in range(1, len(A)+1):
          EditDIstTD(i,j)
  return d[-1][-1]


def looking_for_cluster(p_new_triplets, optimal_clusters, treshold ):
  nearest_cluster = []
  for triplet in p_new_triplets:
    if triplet in optimal_clusters.keys():
      print('found cluster ', triplet) #we have the exact match in triplets buncg
      nearest_cluster.append(triplet)
    elif triplet not in optimal_clusters.keys():
      optimal_array = np.array(list(optimal_clusters.keys()),dtype=object)
      triplet_array = np.array(triplet,dtype=object)
      #loook for triplets that have thr same time Tj and time partition
      triplets_to_check = optimal_array[list(map(bool, np.prod(optimal_array[:,:-1] ==triplet_array[:-1], axis =1)))]

      for trips in triplets_to_check:
        A = trips[2]
        B = triplet[2]
        d = [[2147483646 for j in range(len(A)+1)]for i in range(len(B)+1)]
        if len(A) != len(B):
          print('here')
          if (A in B) or (B in A):
            print('found subcluster' , trips)
            nearest_cluster.append(trips)
          else:
            if Levenshtein(A,B)/ max(len(A), len(B)) < treshold:
              print('found closest cluster', trips)
              nearest_cluster.append(trips)
            else:
              print('no closests cluster for ', triplet)
        elif len(A) == len(B):
          if Levenshtein(A,B)/len(B) < treshold:
            print('found closest cluster', trips)
          else:
            print('no closests cluster for ', triplet)
  return nearest_cluster


def arr_func(events, T_j, delta_T, fws_func):
  events_fws = dict()
  for p_k in events:
    
    fws_val =  fws_func(p_k, delta_T)
    # fws_val = hs.join([str(el) for el in fws_val])
    if type(p_k) == dict:
      # print('check')
      p_k = dict_to_pk(p_k)
    p_k1 = tuple(p_k)
    if p_k1 not in events_fws.keys():
      events_fws[p_k1] = []
      events_fws[p_k1].append(fws_val)
    else:
      # print('here')
      events_fws[p_k1].append(fws_val)

  array = []
  for val in events_fws.values():
    # print(val)
    array.append(list(val[0]))
  return array, events_fws


def events_tensor(events_fws):
  keys_list = list(events_fws.keys())
  full_tensor_batch = torch.tensor([], dtype=torch.float32)
  for key in keys_list:
  # events_fws.values():
    
    ten = torch.tensor(events_fws[key]).unsqueeze(0)
    # print(ten.shape)
    if ten.shape[1] == 1:
      full_tensor_batch = torch.cat((full_tensor_batch.float(), ten.float()), dim=0)
    else:
      for i in range(ten.shape[1]):
        ten2 = ten[:,i , :].unsqueeze(0)
        full_tensor_batch = torch.cat((full_tensor_batch.float(), ten2.float()), dim=0)
  if len(full_tensor_batch.shape) == 4:
    full_tensor_batch = full_tensor_batch.squeeze(axis=1)
  return full_tensor_batch
