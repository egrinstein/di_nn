""" The idea of having two dataset loaders (SydraDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

from sympy import source
import torch

from complex_neural_source_localization.datasets.sydra_dataset import SydraDataset


class SyntheticSSLDataset(SydraDataset):
    def __init__(self, dataset_dir, is_parameterized=True, complex_parameters=True):
        super().__init__(dataset_dir)

        self.is_parameterized = is_parameterized
        self.complex_parameters = complex_parameters

    def __getitem__(self, index):

        (x, y) = super().__getitem__(index)
        
        mic_coordinates = y["mic_coordinates"][:, :2] # Ignore z axis
        source_coordinates = y["source_coordinates"][:2]
        room_dims = y["room_dims"][:2]
        rt60 = torch.Tensor([y["rt60"]])

        y = {
            "source_coordinates": source_coordinates,
            "normalized_source_coordinates": source_coordinates/room_dims,
            "room_dims": room_dims,
            "rt60": rt60
        }

        if self.is_parameterized:
            if self.complex_parameters:
                parameters = torch.complex(mic_coordinates[:, 0], mic_coordinates[:, 1])
                parameters = torch.stack([parameters, torch.complex(rt60, 0)])
            else:
                parameters = torch.hstack([mic_coordinates.flatten(), rt60])

            x = {
                "signal": x,
                "parameters": parameters
            }

        return (x, y)
