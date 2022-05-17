import torch

from di_nn.utils.model_utilities import(
    get_all_layers
)
from di_nn.di_ssl_net import DISSLNET


def test_get_all_layers():
    model = DISSLNET()

    nm = [i for i in model.named_modules()]
    
    layers = get_all_layers(model)
