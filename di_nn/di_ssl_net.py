import torch.nn as nn

from di_nn.feature_extractors import (
    DEFAULT_STFT_CONFIG,
    DecoupledStftArray
)
from di_nn.utils.conv_block import (
    DEFAULT_CONV_CONFIG
)
from di_nn.utils.metadata import DEFAULT_METADATA_CONFIG
from di_nn.utils.di_crnn import DICRNN


class DISSLNET(nn.Module):
    def __init__(self, n_input_channels=4,
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 init_layers=True,
                 metadata_config=DEFAULT_METADATA_CONFIG,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]
        
        # 2. Metadata configuration
        self.metadata_config = metadata_config
        

        self.is_early_fusion = metadata_config["is_early_fusion"] # If true, signals and metadata will be merged at the input
        self.use_metadata_embedding_layer = metadata_config["use_metadata_embedding_layer"]

        n_metadata = 0
        if metadata_config["use_mic_positions"]:
            n_metadata += n_input_channels*2
        if metadata_config["use_room_dims"]:
            n_metadata += 2
        if metadata_config["use_rt60"]:
            n_metadata += 1

        self.is_metadata_aware = n_metadata > 0

        # 2. Create Short Time Fourier Transform feature extractor
        self.stft_layer = DecoupledStftArray(stft_config)

        # 3. Create DI-NN
        self.dicrnn = DICRNN(n_input_channels, 2, pool_type=pool_type,
                             pool_size=pool_size, kernel_size=kernel_size,
                             conv_layers_config=conv_layers_config, fc_layer_dropout_rate=fc_layer_dropout_rate,
                             activation=activation, init_layers=init_layers,
                             is_early_fusion=metadata_config["is_early_fusion"],
                             use_metadata_embedding_layer=self.use_metadata_embedding_layer,
                             n_metadata=n_metadata)
    
    def forward(self, x):
        if self.is_metadata_aware:
            metadata = x["metadata"]
            x = x["signal"]

        # input: (batch_size, mic_channels, time_steps)
        # 1. Extract STFT of signals
        x = self.stft_layer(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)
        if self.is_early_fusion:
            # extract STFT for metadata as well
            metadata = self.stft_layer(metadata)
            metadata = metadata.transpose(2, 3)

        if self.is_metadata_aware:
            # Repack signal and metadata into dictionary
            x = {
                "metadata": metadata,
                "signal": x
            }

        return self.dicrnn(x)
