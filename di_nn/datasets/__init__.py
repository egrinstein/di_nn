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

    dataset = DistributedSSLDataset(dataset_path,
                            config["model"]["is_metadata_aware"],
                            stack_parameters=stack_parameters,
                            use_room_dims_and_rt60=config["model"]["use_room_dims_and_rt60"],
                            is_early_fusion=config["model"]["is_early_fusion"],
                            metadata_dataset_dir=metadata_path)


    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        num_workers=config["training"]["n_workers"]
    )
