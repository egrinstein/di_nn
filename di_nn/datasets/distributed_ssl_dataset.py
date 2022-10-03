""" The idea of having two dataset loaders (BaseDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

import torch

from di_nn.datasets.base_dataset import BaseDataset


class DistributedSSLDataset(BaseDataset):
    def __init__(self, dataset_dir, is_metadata_aware=True, is_early_fusion=False,
                 stack_parameters=True, use_room_dims_and_rt60=True,
                 metadata_dataset_dir=None,
                 metadata_microphone_std_in_m=0,
                 metadata_rt60_std_in_ms=0):

        self.is_early_fusion = is_early_fusion
        if is_early_fusion:
            if metadata_dataset_dir is None:
                raise ValueError(
                    "If using early fusion, the directory containing precomputed metadata signals must be provided.")
        else:
            metadata_dataset_dir = None            

        super().__init__(dataset_dir, metadata_dir=metadata_dataset_dir)

        self.is_metadata_aware = is_metadata_aware
        self.stack_parameters = stack_parameters
        self.use_room_dims_and_rt60 = use_room_dims_and_rt60

        self.metadata_microphone_std_in_m = metadata_microphone_std_in_m
        self.metadata_rt60_std_in_s = metadata_rt60_std_in_ms/1000 

    def __getitem__(self, index):

        (x, y) = super().__getitem__(index)
        
        mic_coordinates = y["mic_coordinates"][:, :2] # Ignore z axis
        source_coordinates = y["source_coordinates"][:2]
        room_dims = y["room_dims"][:2].unsqueeze(0)
        rt60 = torch.Tensor([y["rt60"]])

        targets = {
            "source_coordinates": source_coordinates,
            "normalized_source_coordinates": source_coordinates/room_dims,
            "room_dims": room_dims,
            "rt60": rt60
        }

        # Simulate measurement impresision for sensibility analysis
        mic_coordinates = _simulate_measurement_imprecision(
                            mic_coordinates, self.metadata_microphone_std_in_m)
        rt60 = _simulate_measurement_imprecision(rt60, self.metadata_rt60_std_in_s)

        if self.is_metadata_aware:
            x = {
                "signal": x
            }

            if self.is_early_fusion:
                x["metadata"] = y["metadata_signals"]
            else: # Late fusion
                if self.stack_parameters:
                    if self.use_room_dims_and_rt60:
                        parameters = [mic_coordinates, room_dims, rt60]
                    else:
                        parameters = [mic_coordinates]
                    x["metadata"] = torch.cat([p.flatten() for p in parameters])
                else:
                    x["mic_coordinates"] = mic_coordinates
                    x["room_dims"] = room_dims
                    x["rt60"] = rt60

        return (x, targets)


def _simulate_measurement_imprecision(measurement, std):
    "Disturb the measurement by adding a gaussian value with standard deviation 'std'"
    #print("old measurement:", measurement)
    measurement += torch.randn(measurement.shape)*std
    #print("new measurement:", measurement)
    return measurement
