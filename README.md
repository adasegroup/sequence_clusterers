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

Summary of our results (purities) is presented in the following table:
| Dataset | Cohortney       | DMHP             | Conv Autoencoder          | Transformer Hawkes Process      | Tslearn Kshape | Tslearn Kmeans          | Tsfresh Kmeans         | Tsfresh Kmeans     |
|------------------|---------------------------|---------------------------|------------------------|---------------------------|------------------|---------------------------|---------------------------|------------------|
| exp\_K2\_C5      | 0.97±0.04             | **1.00±0.00**    | 0.65±0.17          | **1.00±0.00**   | 0.76±0.03    | *0.99±0.00* | 0.52±0.01             | 0.94±0.06    |
| exp\_K3\_C5      | *0.85±0.11* | **0.92±0.04**    | 0.64±0.07          | 0.6±0.01             | 0.53±0.05    | 0.80±0.01             | 0.45±0.00             | 0.78±0.08    |
| exp\_K4\_C5      | **0.90±0.07**    | *0.89±0.14* | 0.45±0.04          | 0.67±0.06             | 0.36±0.04    | 0.81±0.02             | 0.27±0.01             | 0.53±0.04    |
| exp\_K5\_C5      | **0.84±0.09**   | *0.66±0.06* | 0.46±0.02          | 0.57±0.04             | 0.42±0.03    | 0.55±0.01             | 0.32±.01             | 0.63±0.04    |
| sin\_K2\_C5      | **0.99±0.00**    | *0.93±0.15* | 0.89±0.03          | 0.89±0.00             | 0.77±0.10    | *0.93±0.00* | 0.52±0.01             | 0.89±0.04    |
| sin\_K3\_C5      | 0.56±0.07             | **0.95±0.02**    | 0.80±0.01          | 0.82±0.00             | 0.44±0.08    | *0.84±0.09* | 0.53±0.06             | 0.75±0.13    |
| sin\_K4\_C5      | **0.92±0.06**    | *0.81±0.08* | 0.62±0.07          | 0.55±0.00             | 0.50±0.05    | 0.59±0.02             | 0.38±0.01             | 0.67±0.04    |
| sin\_K5\_C5      | **0.92±0.05**    | *0.70±0.03* | 0.47±0.01          | 0.51±0.01             | 0.49±0.04    | 0.58±0.03             | 0.26±0.02             | 0.58±0.05    |
| trunc\_K2\_C5    | **1.00±0.00**    | **1.00±0.00**    | **1.00±0.00** | *0.88±0.17* | 0.75±0.10    | **1.00±0.00**    | 0.85±0.00             | 0.78±0.14    |
| trunc\_K3\_C5    | **0.96±0.01**    | **0.96±0.01**    | 0.59±0.09          | 0.61±0.00             | 0.44±0.06    | *0.80±0.01* | 0.34±0.00             | 0.34±0.00    |
| trunc\_K4\_C5    | **0.99±0.00**    | *0.97±0.05* | 0.75±0.07          | 0.67±0.02             | 0.41±0.08    | 0.85±0.10             | 0.28±0.04             | 0.35±0.12    |
| trunc\_K5\_C5    | **0.94±0.06**    | *0.91±0.08* | 0.64±0.10          | 0.60±0.02             | 0.40±0.03    | 0.76±0.07             | 0.23±0.02             | 0.47±0.14    |
| Age              | **0.34±0.03**    | nan±nan               | 0.31±0.00          | *0.33±0.00* | 0.26±0.00    | 0.26±0.00             | 0.26±0.00             | 0.26±0.00    |
| IPTV             | 0.37±0.01             | nan±nan               | 0.36±0.00          | **0.49±0.00**   | 0.34±0.01    | 0.35±0.02             | *0.38±0.01* | 0.36±0.01    |
| Linkedin         | *0.26±0.06* | nan±nan               | 0.21±0.00          | 0.22±0.01             | 0.20±0.00    | 0.20±0.00             | **0.44±0.02**| 0.22±0.01    |
| ATM              | *0.46±0.03* | nan±nan               | 0.28±0.00          | 0.43±0.00             | 0.31±0.03    | **0.50±0.01**    | 0.29±0.00             | 0.29±$0.00    |
| Nr. of wins      | **10**              | 5                         | 1                      | 2                         | 0                | 2                         | 1                         | 0                |
