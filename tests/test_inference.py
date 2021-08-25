import torch

from neural_tdoa.model import TdoaCrnn10
from datasets.dataset import TdoaDataset


def test_inference():
    model = TdoaCrnn10()
    
    weights = torch.load("tests/fixtures/weights.pth",
                         map_location=torch.device('cpu'))
    #breakpoint()
    model.load_state_dict(weights)
    model.eval()

    dataset = TdoaDataset()

    sample = dataset[0]
    target = sample[1]

    model_output = model(sample[0].unsqueeze(0))

    #breakpoint()