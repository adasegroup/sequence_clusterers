import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src import (
    train_model,
    run_inference,
)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Run different tasks
    if config.task_type == "infer_only":
        run_inference(config)
    elif config.task_type == "train":
        train_model(config)
    else:
        raise Exception(f"Warning {config.task_type} is not supported")


if __name__ == "__main__":
    main()
