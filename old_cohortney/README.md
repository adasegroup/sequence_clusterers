# COHORTNEY: Deep Clustering for Heterogeneous Event Sequences
Here we provide the implementation of COHORTNEY.
The publication is currently under review.

## Data
We provide 17 datasets: 12 synthetic and 5 real-world. All datasets are
in the GDrive: https://drive.google.com/drive/folders/1xSjHx7SQDEefgCuAeP21NLOabIpL13XH?usp=sharing
- data/[sin_,trunc_]Kx_C5 - synthetic datasets
- data/[Age,Linkedin,IPTV,ATM,Booking] - real world datasets
All the datasets should be saved to the data/ folder.

##  Method
We use an LSTM-based model to estimate the intensity as
a piecewise constant function. The model is in 'models/LSTM.py'.

### Highlights

The ```get_partition``` function in 'utils/data_preprocessor.py' preprocesses
point processes to a format that is suitable for the LSTM

The file 'data\trainers.py' consists of the Trainer class. It conducts the model training

## Starting the experiments
To start the experiments, one needs to run the following command (e.g. for K5_C5
dataset):

```
python run.py --path_to_files data/K5_C5 --n_steps 128 --n_clusters 1
--true_clusters 5 --upper_bound_clusters 10 --random_walking_max_epoch 40
--n_classes 5 --lr 0.1 --lr_update_param 0.5 --lr_update_tol 25 --n_runs 5
--save_dir K5_C5 --max_epoch 50 --max_m_step_epoch 10 --min_lr 0.001
--updated_lr 0.001 --max_computing_size 800 --device cuda:0
```

To run all the experiments run 'start.sh' script:
```
./start.sh
```

All the results and the parameters are stored in 'experiments/[save_dir]' folder:
- 'experiments/[save_dir]/args.json' has the parameters.
- 'experiments/[save_dir]/last_model.pt' has the model.

## The results

| **Dataset** | **COHORTNEY**       | **DMHP**               | **Soft**  | **K**    | **K-means**        | **K-means**  | **GMM**      |
|------------------|---------------------------|------------------------------|-----------------|----------------|-------------------------|--------------------|--------------------|
|                  | **(ours)**           |  | **DTW**    | **Shape** | **partitions**     |**tsfresh**   | **tsfresh**   |
| K2\_C5           | **1.00 &pm; 0.00**    | 0.91 &pm; 0.00                | 0.50 &pm; 0.0    | 0.50 &pm; 0.0   | 0.89  &pm;  0.0          | 0.92 | 0.92 |
| K3\_C5           | **0.99 &pm; 0.00**    | 0.66 &pm; 0.00                | 0.33 &pm; 0.0    | 0.33 &pm; 0.0   | 0.52  &pm;  0.0          | 0.72             | 0.89 |
| K4\_C5           | **0.98 &pm; 0.06**    | 0.80 &pm; 0.08                | 0.25 &pm; 0.0    | 0.25 &pm; 0.0   | 0.60  &pm;  0.0          | 0.86 | 0.76             |
| K5\_C5           | **0.94 &pm; 0.10**    | 0.58 &pm; 0.03                | -             | 0.20 &pm; 0.0   | 0.58  &pm;  0.0          | 0.76 | **0.94**    |
| sin\_K2\_C5      | **0.99 &pm; 0.01**    | 0.98 &pm; 0.05    | 0.50 &pm; 0.0    | 0.50 &pm; 0.0   | 0.93  &pm;  0.0          | 0.52             | 0.96             |
| sin\_K3\_C5      | **0.99 &pm; 0.01**    | 0.98 &pm; 0.00    | 0.33 &pm; 0.0    | 0.33 &pm; 0.0   | 0.85  &pm;  0.0          | 0.57             | 0.87             |
| sin\_K4\_C5      | **0.93 &pm; 0.04**    | 0.58 &pm; 0.06                | 0.25 &pm; 0.0    | 0.25 &pm; 0.0   | 0.51  &pm;  0.0          | 0.38             | 0.68 |
| sin\_K5\_C5      | **0.92 &pm; 0.05**    | 0.75 &pm; 0.05    | 0.20 &pm; 0.0    | 0.20 &pm; 0.0   | 0.56  &pm;  0.0          | 0.30             | 0.69             |
| trunc\_K2\_C5    | **1.00 &pm; 0.00**    | **1.00 &pm; 0.00**       | 0.50 &pm; 0.0    | 0.50 &pm; 0.0   | **1.00  &pm;  0.0** | 0.85 | 0.85** |
| trunc\_K3\_C5    | **0.96 &pm; 0.01**    | 0.67 &pm; 0.00    | 0.33 &pm; 0.0    | 0.33 &pm; 0.0   | 0.45  &pm;  0.0          | 0.99             | 0.99             |
| trunc\_K4\_C5    | **0.99 &pm; 0.00**    | **0.99 &pm; 0.00**       | 0.25 &pm; 0.0    | 0.25 &pm; 0.0   | 0.75  &pm;  0.0          | **0.99**    | **0.99**    |
| trunc\_K5\_C5    | 0.95 &pm; 0.04 | 0.88 &pm; 0.09                | 0.20 &pm; 0.0    | 0.20 &pm; 0.0   | 0.44  &pm;  0.0          | **0.99**    | **0.99**    |
| IPTV             | 0.37 &pm; 0.01             | 0.34 &pm; 0.03                | 0.32 &pm; 0.0    | 0.32 &pm; 0.0   | 0.34  &pm;  0.0          | **0.80**    | 0.44 |
| Age              | 0.36 &pm; 0.01             | 0.38 &pm; 0.01                | -             | -            | 0.35  &pm;  0.0          | **0.99**    | 0.41 |
| Linkedin         | 0.34 &pm; 0.06             | 0.31 &pm; 0.01                | 0.20 &pm; 0.0    | 0.20 &pm; 0.0   | 0.20  &pm;  0.0          | **0.46**    | 0.42 |
| ATM              | 0.69 &pm;  0.05            | 0.64 &pm;  0.02               | 0.14  &pm;  0.0  | 0.14  &pm;  0.0 | -                     | **0.99**    | **0.99**    |
| Booking.com      | 0.54 &pm;  0.08            | -                          | 0.33 &pm; 0.0    | 0.33 &pm; 0.0   | -                     | **0.99**    | **0.99**    |
| Nr. of wins      | **11**             | 2                          | 0             | 0            | 1                     | 7                | 5                |
