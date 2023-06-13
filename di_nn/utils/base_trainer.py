import pickle
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import (
    ModelCheckpoint, TQDMProgressBar
)
from pytorch_lightning import loggers as pl_loggers

from di_nn.utils.model_utilities import merge_list_of_dicts

SAVE_DIR = "logs/"

class BaseTrainer(pl.Trainer):
    def __init__(self, lightning_module, n_epochs, use_checkpoint_callback=True):

        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            save_last=True,
            filename='weights-{epoch:02d}-{validation_loss:.2f}',
            save_weights_only=True
        )

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=SAVE_DIR)
        csv_logger = pl_loggers.CSVLogger(save_dir=SAVE_DIR)

        callbacks=[] # feature_map_callback],
        if use_checkpoint_callback:
            callbacks.append(checkpoint_callback)

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        # Don't use MPS, which is a nightmare 

        super().__init__(
            max_epochs=n_epochs,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            accelerator=accelerator,
            log_every_n_steps=25
        )
        
        self._lightning_module = lightning_module


class BaseLightningModule(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    def __init__(self, model, loss,
                 log_step=50):
        super().__init__()

        self.model = model
        self.loss = loss

        self.log_step = log_step
        self.outputs = {
            "train": [],
            "validation": [],
            "test": []
        }

    def _step(self, batch, batch_idx, log_model_output=False,
              log_labels=False, epoch_type="train"):

        x, y = batch

        # 1. Compute model output and loss
        output = self.model(x)
        loss = self.loss(output, y, mean_reduce=False)

        output_dict = {
            "loss_vector": loss
        }

        # TODO: Add these to a callback
        # 2. Log model output
        if log_model_output:
            output_dict["model_output"] = output
        # 3. Log ground truth labels
        if log_labels:
            output_dict.update(y)

        output_dict["loss"] = output_dict["loss_vector"].mean()
        output_dict["loss_vector"] = output_dict["loss_vector"].detach().cpu()
        
        # 4. Log step metrics
        self.log("loss_step", output_dict["loss"], on_step=True, prog_bar=False)

        self.outputs[epoch_type].append(output_dict)

        return output_dict

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, epoch_type="train")
  
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, epoch_type="validation",
                          log_model_output=False, log_labels=False)
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, epoch_type="test",
                          log_model_output=False, log_labels=False)
    
    def _epoch_end(self, epoch_type="train", save_pickle=False):
        # 1. Compute epoch metrics
        outputs = merge_list_of_dicts(self.outputs[epoch_type])
        epoch_stats = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            f"{epoch_type}_std": outputs["loss"].std()
        }

        # 2. Log epoch metrics
        for key, value in epoch_stats.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        # 3. Save complete epoch data on pickle
        if save_pickle:
            pickle_filename = f"{epoch_type}.pickle"
            with open(pickle_filename, "wb") as f:
                pickle.dump(outputs, f)

        return epoch_stats
    
    def on_train_epoch_end(self):
        self._epoch_end()

    def on_validation_epoch_end(self):
        self._epoch_end(epoch_type="validation")

    def on_test_epoch_end(self):
        self._epoch_end(epoch_type="test", save_pickle=True)

    def forward(self, x):
        return self.model(x)
        
    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test, ckpt_path="best"):
        super().test(self.model, dataset_test, ckpt_path=ckpt_path)
