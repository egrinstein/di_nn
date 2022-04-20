import torch

from complex_neural_source_localization.utils.model_utilities import(
    get_all_layers
)
from complex_neural_source_localization.model import SSLNET


def test_get_all_layers():
    model = SSLNET()

    nm = [i for i in model.named_modules()]
    
    layers = get_all_layers(model)
