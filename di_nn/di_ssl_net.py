import torch.nn as nn

from di_nn.feature_extractors import (
    DEFAULT_STFT_CONFIG,
    DecoupledStftArray
)
from di_nn.utils.conv_block import (
    DEFAULT_CONV_CONFIG
)
from di_nn.utils.di_crnn import DICRNN


class DISSLNET(nn.Module):
    def __init__(self, n_input_channels=4,
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 init_layers=True,
                 is_metadata_aware=False,
                 use_room_dims_and_rt60=False,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]
        self.is_metadata_aware = is_metadata_aware # Parameterized Neural Network:
                                           # concatenate the microphone coordinates to the features before
                                           # feeding them to the fully connected layers
        self.use_room_dims_and_rt60 = use_room_dims_and_rt60
        n_metadata = n_input_channels*2
        if use_room_dims_and_rt60:
            n_metadata += 3 # 3 => room width, room length, reverberation time
    
        # 2. Create Short Time Fourier Transform feature extractor
        self.stft_layer = DecoupledStftArray(stft_config)

        # 3. Create DI-NN
        self.dicrnn = DICRNN(n_input_channels, 2, pool_type=pool_type,
                             pool_size=pool_size, kernel_size=kernel_size,
                             conv_layers_config=conv_layers_config, fc_layer_dropout_rate=fc_layer_dropout_rate,
                             activation=activation, init_layers=init_layers, is_metadata_aware=is_metadata_aware,
                             n_metadata=n_metadata)
    
    def forward(self, x):
        if self.is_metadata_aware:
            parameters = x["parameters"]
            x = x["signal"]

        # input: (batch_size, mic_channels, time_steps)
        # 1. Extract STFT of signals
        x = self.stft_layer(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)

        if self.is_metadata_aware:
            # Repack signal and parameters into dictionary
            x = {
                "parameters": parameters,
                "signal": x
            }

        return self.dicrnn(x)
