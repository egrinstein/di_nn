""" The idea of having two dataset loaders (BaseDataset and Dataset)
is because in theory the random generator implemented in Sydra could be reused for different
applications using two microphones, while this generator is specific for this application.
"""

import torch

from di_nn.datasets.base_dataset import BaseDataset
from di_nn.utils.metadata import DEFAULT_METADATA_CONFIG


class DistributedSSLDataset(BaseDataset):
    def __init__(self, dataset_dir,
                 stack_parameters=True,
                 metadata_dataset_dir=None,
                 metadata_microphone_std_in_m=0,
                 metadata_rt60_std_in_ms=0,
                 metadata_config=DEFAULT_METADATA_CONFIG):

        self.metadata_config = metadata_config
        self.is_early_fusion = metadata_config["is_early_fusion"]
        
        if self.is_early_fusion:
            if metadata_dataset_dir is None:
                raise ValueError(
                    "If using early fusion, the directory containing precomputed metadata signals must be provided.")
        else:
            metadata_dataset_dir = None            

        super().__init__(dataset_dir, metadata_dir=metadata_dataset_dir)

        self.use_mic_positions = metadata_config["use_mic_positions"]
        self.use_room_dims = metadata_config["use_room_dims"]
        self.use_rt60 = metadata_config["use_rt60"]

        self.is_metadata_aware = self.use_mic_positions or self.use_room_dims or self.use_rt60

        self.stack_parameters = stack_parameters

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
                metadata = {}
                if self.use_mic_positions:
                    metadata["mic_coordinates"] = mic_coordinates
                if self.use_room_dims:
                    metadata["room_dims"] = room_dims
                if self.use_rt60:
                    metadata["rt60"] = rt60

                if self.stack_parameters:
                    x["metadata"] = torch.cat([
                        p.flatten() for p in metadata.values()
                    ])
                else:
                    x["metadata"] = metadata

        return (x, targets)


def _simulate_measurement_imprecision(measurement, std):
    "Disturb the measurement by adding a gaussian value with standard deviation 'std'"
    #print("old measurement:", measurement)
    measurement += torch.randn(measurement.shape)*std
    #print("new measurement:", measurement)
    return measurement
