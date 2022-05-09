""" The idea of having two dataset loaders (SydraDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

import torch

from complex_neural_source_localization.datasets.sydra_dataset import SydraDataset


class DistributedSSLDataset(SydraDataset):
    def __init__(self, dataset_dir, is_parameterized=True, complex_parameters=True, stack_parameters=True):
        super().__init__(dataset_dir)

        self.is_parameterized = is_parameterized
        self.complex_parameters = complex_parameters
        self.stack_parameters = stack_parameters

    def __getitem__(self, index):

        (x, y) = super().__getitem__(index)
        
        mic_coordinates = y["mic_coordinates"][:, :2] # Ignore z axis
        source_coordinates = y["source_coordinates"][:2]
        room_dims = y["room_dims"][:2].unsqueeze(0)
        rt60 = torch.Tensor([y["rt60"]])

        y = {
            "source_coordinates": source_coordinates,
            "normalized_source_coordinates": source_coordinates/room_dims,
            "room_dims": room_dims,
            "rt60": rt60
        }

        if self.is_parameterized:
            if self.complex_parameters:
                mic_coordinates = torch.complex(mic_coordinates[:, 0], mic_coordinates[:, 1])
                room_dims = torch.complex(room_dims[0], room_dims[1])
                rt60 = torch.complex(rt60, 0)

            x = {
                "signal": x
            }

            if self.stack_parameters:
                if self.complex_parameters:
                    x["parameters"] = torch.stack([mic_coordinates, room_dims, rt60])
                else:
                    x["parameters"] = torch.cat([
                        mic_coordinates.flatten(), room_dims.flatten(), rt60])
            else:
                x["mic_coordinates"] = mic_coordinates
                x["room_dims"] = room_dims
                x["rt60"] = rt60

        return (x, y)
