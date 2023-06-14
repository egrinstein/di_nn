import torch

from .distributed_ssl_dataset import DistributedSSLDataset


def create_torch_dataloaders(config):
    return (
        create_torch_dataloader(config, "training"),
        create_torch_dataloader(config, "validation"),
        create_torch_dataloader(config, "test"),
    )


def create_torch_dataloader(config, mode, stack_parameters=True):
    if mode == "training":
        dataset_path = config["dataset"]["training_dataset_dir"]
        metadata_path = config["dataset"]["metadata_training_dataset_dir"]
        shuffle = True
    elif mode == "validation":
        dataset_path = config["dataset"]["validation_dataset_dir"]
        metadata_path = config["dataset"]["metadata_validation_dataset_dir"]
        shuffle = False
    elif mode == "test":
        dataset_path = config["dataset"]["test_dataset_dir"]
        metadata_path = config["dataset"]["metadata_test_dataset_dir"]
        shuffle = False

    if config["dataset"]["disturb_metadata_on_test_only"] and mode != "test":
        metadata_microphone_std_in_m = 0
        metadata_rt60_std_in_ms = 0
    else:
        metadata_microphone_std_in_m = config["dataset"]["metadata_microphone_std_in_m"]
        metadata_rt60_std_in_ms = config["dataset"]["metadata_rt60_std_in_ms"]

    metadata_config = config["model"]["metadata_config"]

    dataset = DistributedSSLDataset(
        dataset_path,
        stack_parameters=stack_parameters,
        metadata_dataset_dir=metadata_path,
        metadata_microphone_std_in_m=metadata_microphone_std_in_m,
        metadata_rt60_std_in_ms=metadata_rt60_std_in_ms,
        metadata_config=metadata_config
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        num_workers=config["training"]["n_workers"]
    )
