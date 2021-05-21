# Cohortney

Framework for Deep Clustering of Heterogeneous Event Sequences.

For conv1d AutoEncoder over sequences
```shell script
 PYTONPATH='.' python3 run.py aux_module=cae +task_type=cae
```

Train deep clustering over sequences
```shell script
PYTONPATH='.' python3 run.py aux_module=deep_cluster +task_type=deep_clustering
```

In order to reproduce experiments, the one firstly should configure:
```
- param. "data_dir" in ~/configs/config.yaml responsible for the dataset to be downloaded and preprocessed (based on in ~/blob/main/src/utils/__init__.py); 
- hyperparams. of corresponding method in ~/configs/aux_module/;
- training parameters (CPU\GPU) in ~/configs/trainer/default.yaml as well as NN-related training parameter (number of epochs);
```


Datasets:
Linkedin dataset, 
Synthetic Hawkes processes realizations

The dataset is taken from original [repo](https://github.com/VladislavZh/pp_clustering)


Overview os the project:
The project is the implementations on the methods to cluster sequences taken from two Github repositories  in a single repository with the unified  API and code structure that follows best practises of formatting. In the project we focus on the only working methods among propoced, namely DeepCluster and CAE (both over claasical Pure Chortney method) and do not include in the repository the implementations from parent repository that do not work based on execution manuals provided there. The methods aare refactored with use if Pytorch Lightning framework so that execution of the methods is handy and fast with very deep range of configurable parameters both for the experiments and of hardware usage. 

The validation results are stored by the link: ``` https://github.com/adasegroup/cohortney/tree/main/results(logs) ``` 

More detailed information about the project may be found in the recent report(s).


