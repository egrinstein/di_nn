import shutil
import torch

from neural_tdoa.metrics import Loss
from neural_tdoa.model import TdoaCrnn10
from datasets.dataset import TdoaDataset


def test_neural_tdoa_loss():
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)
    
    loss_fn = Loss()
    model = TdoaCrnn10()

    dataset = TdoaDataset(n_samples=1, dataset_dir=temp_dataset_path)

    sample = dataset[0]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))
    
    loss = loss_fn(model_output, torch.Tensor([[target]]))

