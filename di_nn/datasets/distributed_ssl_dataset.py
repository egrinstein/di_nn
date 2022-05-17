""" The idea of having two dataset loaders (SydraDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

import torch

from di_nn.datasets.sydra_dataset import SydraDataset


class DistributedSSLDataset(SydraDataset):
    def __init__(self, dataset_dir, is_metadata_aware=True,
                 stack_parameters=True, use_room_dims_and_rt60=True):
        super().__init__(dataset_dir)

        self.is_metadata_aware = is_metadata_aware
        self.stack_parameters = stack_parameters
        self.use_room_dims_and_rt60 = use_room_dims_and_rt60

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

        if self.is_metadata_aware:
            x = {
                "signal": x
            }

            if self.stack_parameters:
                if self.use_room_dims_and_rt60:
                    parameters = [mic_coordinates, room_dims, rt60]
                else:
                    parameters = [mic_coordinates]

                x["parameters"] = torch.cat([p.flatten() for p in parameters])

            else:
                x["mic_coordinates"] = mic_coordinates
                x["room_dims"] = room_dims
                x["rt60"] = rt60

        return (x, y)
