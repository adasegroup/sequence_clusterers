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



Datasets:
Linkedin dataset
Synthetic Hawkes processes realizations

The dataset is taken from original [repo](https://github.com/VladislavZh/pp_clustering)


Overview os the project:
The project is the implementations on the methods to cluster sequences taken from two Github repositories  in a single repository with the unified  API and code structure that follows best practises of formatting. The methods aare refactored with use if Pytorch Lightning framework so that execution of the methods is handy and fast with very deep range of configurable parameters both for the experiments and of hardware usage. 

The validation results are stored by the link: ``` https://github.com/adasegroup/cohortney/tree/main/results(logs) ``` 

More detailed information about the project may be found in the recent report(s).

The current structure of the repository:

```
├── README.md
├── configs
│   ├── callbacks
│   │   └── default.yaml
│   ├── config.yaml
│   ├── datamodule
│   │   └── default.yaml
│   ├── experiment
│   │   └── default.yaml
│   ├── logger
│   │   └── default.yaml
│   ├── model
│   │   └── default.yaml
│   └── trainer
│       └── default.yaml
├── reports
│   └── cohortney_report_1.pdf
├── requirements.txt
├── reviews
│   ├── peer1_cohortney_mmdf.pdf
│   ├── peer1_cohortney_oms.pdf
│   └── peer1_cohortney_prophet.pdf
├── run.py
└── src
    ├── __init__.py
    ├── dataset
    │   ├── __init__.py
    │   └── random_seq.py
    ├── dataset_generator.py
    ├── model
    │   └── single_pp_cohortney.py
    ├── networks
    │   ├── losses.py
    │   └── lstm_pp.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── base.py
        ├── datamodule.py
        ├── metrics.py
        └── net.py
```
