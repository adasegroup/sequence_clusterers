import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    from src.train import train
    from src.utils import *
    # Train model
    return train(config)


if __name__ == "__main__":
    main()