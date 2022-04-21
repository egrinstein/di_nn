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

        y = {
            "source_coordinates": source_coordinates,
            "normalized_source_coordinates": source_coordinates/room_dims,
            "room_dims": room_dims
        }

        if self.is_parameterized:
            parameters = torch.vstack([mic_coordinates, room_dims.unsqueeze(0)])
            if self.complex_parameters:
                parameters = torch.complex(parameters[:, 0], parameters[:, 1])
            else:
                parameters = parameters.flatten()

            x = {
                "signal": x,
                "parameters": parameters
            }

        return (x, y)
