import torch

from .dcase_2019_task3_dataset import DCASE2019Task3Dataset
from .distributed_ssl_dataset import DistributedSSLDataset

DATASET_NAME_TO_CLASS_MAP = {
    "distributed_ssl": DistributedSSLDataset,
    "dcase_2019_task3": DCASE2019Task3Dataset
}


def create_torch_dataloaders(config):
    return (
        create_torch_dataloader(config, "training"),
        create_torch_dataloader(config, "validation"),
        create_torch_dataloader(config, "test"),
    )


def create_torch_dataloader(config, mode, stack_parameters=True):
    if mode == "training":
        dataset_path = config["dataset"]["training_dataset_dir"]
        shuffle = True
    elif mode == "validation":
        dataset_path = config["dataset"]["validation_dataset_dir"]
        shuffle = False
    elif mode == "test":
        dataset_path = config["dataset"]["test_dataset_dir"]
        shuffle = False

    dataset_class = DATASET_NAME_TO_CLASS_MAP[config["dataset"]["name"]]
    dataset = dataset_class(dataset_path,
                            config["model"]["is_metadata_aware"],
                            stack_parameters=stack_parameters,
                            use_room_dims_and_rt60=config["model"]["use_room_dims_and_rt60"])


    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
        num_workers=config["training"]["n_workers"]
    )
