# Cohortney

Framework for Deep Clustering of Heterogeneous Event Sequences.

Datasets:
IPTV dataset
Synthetic Hawkes processes realizations

The dataset is taken from original [repo](https://github.com/VladislavZh/pp_clustering)


Overall tasks of the project:
1. Take the implementations on the two Github repositories to cluster sequences and merge them into a single repository: Cohortney, CAE, pure Cohortney, DMHP, DeepCluster, Optimal Transport.
2. Design a standard API for all of these methods and a code structure that follows best practices using the Google Python Style Guide and formatting it with Black.
3. Refactor the six methods with PyTorch Lightning as the main framework for the library.
4. Introduce Ray to optimize all hyperparameters. PyTorch Lightning offers the necessary hooks for this.
5. The execution should be fast, so either you use GPU acceleration or Ray (in addition to Ray for hyperparameter optimization). It also enables parallelization of the code whenever GPU acceleration is not possible.
6. test coverage of 50% of the codebase.
7. documentation for the API using readthedocs.
8. examples on how to run each method.
9. reproduce the experiment section of COHORTNEY with your new library on all six methods.
10. pip package
