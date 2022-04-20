""" The idea of having two dataset loaders (SydraDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

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

        y = {
            "mic_coordinates": mic_coordinates,
            "source_coordinates": source_coordinates
        }

        if self.is_parameterized:
            if self.complex_parameters:
                mic_coordinates = torch.complex(mic_coordinates[:, 0],
                                                mic_coordinates[:, 1])
            else:
                mic_coordinates = mic_coordinates.flatten()

            x = {
                "signal": x,
                "mic_coordinates": mic_coordinates
            }

        return (x, y)
