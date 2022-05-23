import hydra

from omegaconf import DictConfig

from di_nn.datasets import create_torch_dataloaders
from di_nn.trainer import DISSLNETTrainer


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_train, dataset_val, dataset_test = create_torch_dataloaders(config)
    
    trainer = DISSLNETTrainer(config)

    trainer.fit(dataset_train, val_dataloaders=dataset_val)
    trainer.test(dataset_test)


if __name__ == "__main__":
    train()
