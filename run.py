import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src import train, cae_train, deep_cluster_train
from src.utils.datamodule import download_dataset


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Download specific dataset
    download_dataset(config.data_dir, config.data_dir.split('/')[-1])

    # Train model
    if config.task_type == 'deep_clustering':
        deep_cluster_train(config)
    elif config.task_type == 'cae':
        cae_train(config)
    elif config.task_type == 'multi_pp':
        train(config)
    else:
        raise Exception(f"Warning {config.task_type} is not supported")


if __name__ == "__main__":
    main()
