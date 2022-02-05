# Sequence Clusterers

Framework of Methods for Clustering of Heterogeneous Event Sequences.

To train model:
```shell script
python3 run.py --config_name config_thp.yaml +task_type=train
```

To use pretrained model for inference only:
```shell script
python3 run.py --config_name config_ts.yaml +task_type=infer_only
```

To reproduce experiments, one should edit several config files:
```
- param. "data_name" in ~/configs/config_name.yaml to choose corresponging dataset; 
- hyperparams of corresponding method in ~/configs/model/name_of_method.yaml;
- training parameters (device, number of epochs, etc.) in ~/configs/trainer/default.yaml;
- hyperparameters of data preprocessing (max sequence length, batch size, etc.) in ~/configs/datamodule/name_of_datamodule.yaml;
```


Datasets:
LinkedIn, Age, ATM, IPTV, Synthetic Hawkes processes realizations

The datasets are taken from [cloud drive](https://drive.google.com/drive/folders/1xSjHx7SQDEefgCuAeP21NLOabIpL13XH)


Overview of the project:
The project is the implementation of sequences clusterization methods using the common API and code structure that follows best practices of formatting. In the project we focus on proprietary method (aka Cohortney) and several baselines, including Convolutional Autoencoder, Transformer Hawkes Process, TsFresh and TsLearn feature extractors. The methods are refactored to fit Pytorch Lightning framework. 

