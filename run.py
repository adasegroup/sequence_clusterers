import hydra
from omegaconf import DictConfig

from src import train


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Train model
    return train(config)


if __name__ == "__main__":
    main()