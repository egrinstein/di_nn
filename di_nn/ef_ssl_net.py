import torch
import torch.nn as nn

from di_nn.feature_extractors import (
    DEFAULT_STFT_CONFIG,
    DecoupledStftArray
)
from di_nn.utils.conv_block import (
    DEFAULT_CONV_CONFIG
)
from di_nn.utils.di_crnn import DICRNN


class EFSSLNET(nn.Module):
    def __init__(self, n_input_channels=4,
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 init_layers=True,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = 2*n_input_channels # One extra metadata channel per input
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]
        
        # 2. Create Short Time Fourier Transform feature extractor
        self.stft_layer = DecoupledStftArray(stft_config)

        # 3. Create DI-NN
        self.dicrnn = DICRNN(n_input_channels, 2, pool_type=pool_type,
                             pool_size=pool_size, kernel_size=kernel_size,
                             conv_layers_config=conv_layers_config, fc_layer_dropout_rate=fc_layer_dropout_rate,
                             activation=activation, init_layers=init_layers, is_metadata_aware=False)
    
    def forward(self, x):
        parameters = x["parameters"]
        x = x["signal"]

        x = torch.cat([x, parameters], dim=1)

        # input: (batch_size, mic_channels, time_steps)
        # 1. Extract STFT of signals
        x = self.stft_layer(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)

        return self.dicrnn(x)
