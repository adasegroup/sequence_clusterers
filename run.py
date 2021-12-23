import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src import (
    train,
    cae_train,
    deep_cluster_train,
    thp_train,
    tslearn_infer,
    tsfresh_infer,
)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Train model
    if config.task_type == "deep_clustering":
        deep_cluster_train(config)
    elif config.task_type == "cae":
        cae_train(config)
    elif config.task_type == "thp":
        thp_train(config)
    elif config.task_type == "multi_pp":
        train(config)
    elif config.task_type == "tslearn":
        tslearn_infer(config)
    elif config.task_type == "tsfresh":
        tsfresh_infer(config)
    else:
        raise Exception(f"Warning {config.task_type} is not supported")


if __name__ == "__main__":
    main()
