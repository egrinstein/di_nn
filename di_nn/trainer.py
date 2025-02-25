from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR

from di_nn.di_ssl_net import DISSLNET
from di_nn.loss import LOSS_NAME_TO_CLASS_MAP
from di_nn.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DISSLNETTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DISSLNETLightniningModule(config)
        super().__init__(lightning_module,
                         config["training"]["n_epochs"])

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders, ckpt_path="best"):
        super().test(self._lightning_module, test_dataloaders, ckpt_path=ckpt_path)


class DISSLNETLightniningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DISSLNET
    """

    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config

        n_sources = self.config["dataset"]["n_max_sources"]

        stft_config = {
            "n_fft": config["model"]["n_fft"],
            "use_onesided_fft": config["model"]["use_onesided_fft"]
        }

        model = DISSLNET(
            n_sources=n_sources,
            stft_config=stft_config,
            **config["model"]
        )

        loss = LOSS_NAME_TO_CLASS_MAP[self.config["model"]["loss"]]()

        super().__init__(model, loss,
                         checkpoint_path=config["training"]["checkpoint_path"])

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]
