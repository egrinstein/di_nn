import torch
import torch.nn as nn

from complex_neural_source_localization.feature_extractors import (
    DEFAULT_STFT_CONFIG,
    FEATURE_NAME_TO_CLASS_MAP
)
from complex_neural_source_localization.utils.conv_block import (
    DEFAULT_CONV_CONFIG, ConvBlock
)
from complex_neural_source_localization.utils.complexPyTorch.complexLayers import (
    ComplexGRU, ComplexLinear, ComplexPReLU, ComplexReLU
)
from complex_neural_source_localization.utils.model_utilities import init_gru, init_layer


class SSLNET(nn.Module):
    def __init__(self, output_type="scalar", n_input_channels=4, n_sources=2,
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 feature_type="stft",
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 stft_config=DEFAULT_STFT_CONFIG,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 is_fully_complex=False,
                 init_real_layers=True,
                 is_parameterized=False,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.output_type = output_type
        self.n_input_channels = n_input_channels
        self.n_sources = n_sources
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]
        self.is_fully_complex = is_fully_complex
        self.is_parameterized = is_parameterized # Parameterized Neural Network:
                                           # concatenate the microphone coordinates to the features before
                                           # feeding them to the fully connected layers
        print(conv_layers_config)

        # 2. Create feature extractor
        self.feature_extractor = self._create_feature_extractor(feature_type, stft_config)

        # 3. Create convolutional blocks
        self.conv_blocks = self._create_conv_blocks(conv_layers_config, init_weights=init_real_layers)

        # 4. Create recurrent block
        self.rnn = self._create_rnn_block()

        # 5. Create linear block
        self.fully_connected = self._create_fully_connected_block(n_sources, fc_layer_dropout_rate)

        # If using a real valued rnn, initialize gru and fc layers
        if not is_fully_complex and init_real_layers:
            init_gru(self.rnn)
            init_layer(self.fully_connected)
    
    def forward(self, x):
        if self.is_parameterized:
            parameters = x["parameters"]
            x = x["signal"]

        # input: (batch_size, mic_channels, time_steps)
        # 1. Extract STFT of signals
        x = self.feature_extractor(x)
        if x.is_complex() and not self.is_fully_complex:
            x = complex_to_real(x)
        # (batch_size, mic_channels, n_freqs, stft_time_steps)
        x = x.transpose(2, 3)
        # (batch_size, mic_channels, stft_time_steps, n_freqs)

        # 2. Extract features using convolutional layers
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # (batch_size, feature_maps, time_steps, n_freqs)

        # 3. Average across all frequencies
        x = torch.mean(x, dim=3)
        # (batch_size, feature_maps, time_steps)

        # Preprocessing for RNN
        x = x.transpose(1,2)
        # (batch_size, time_steps, feature_maps):
        
        # 4. Use features as input to RNN
        (x, _) = self.rnn(x)
        # (batch_size, time_steps, feature_maps):
        # Average across all time steps
        x = torch.mean(x, dim=1)
        # (batch_size, feature_maps)

        if self.is_parameterized:
            # Concatenate parameters before sending to fully connected layer
            x = torch.cat([x, parameters], dim=1)
            # (batch_size, feature_maps + n_parameters)
        
        # 5. Fully connected layer
        x = self.fully_connected(x)
        # (batch_size, class_num)

        if x.is_complex():
            x = complex_to_real(x)
        return x

    def _create_feature_extractor(self, feature_type, stft_config):
        if feature_type == "cross_spectra":
            self.n_feature_extractor_channels = sum(range(self.n_input_channels + 1))
        else:
            self.n_feature_extractor_channels = self.n_input_channels

        feature_extractor = FEATURE_NAME_TO_CLASS_MAP[feature_type](stft_config)
        
        if feature_extractor.is_complex and not self.is_fully_complex:
            # channels will be separated into real and imaginary components
            self.n_feature_extractor_channels *= 2
        
        return feature_extractor

    def _create_conv_blocks(self, conv_layers_config, init_weights):
        
        conv_blocks = [
            ConvBlock(self.n_feature_extractor_channels, conv_layers_config[0]["n_channels"],
                      block_type=conv_layers_config[0]["type"],
                      dropout_rate=conv_layers_config[0]["dropout_rate"],
                      pool_size=self.pool_size,
                      activation=self.activation,
                      kernel_size=self.kernel_size,
                      is_complex=self.is_fully_complex,
                      init=init_weights),
        ]

        for i, config in enumerate(conv_layers_config[1:]):
            last_layer = conv_blocks[-1]
            in_channels = last_layer.out_channels
            conv_blocks.append(
                ConvBlock(in_channels, config["n_channels"],
                          block_type=config["type"],
                          dropout_rate=config["dropout_rate"],
                          pool_size=self.pool_size,
                          activation=self.activation,
                          kernel_size=self.kernel_size,
                          is_complex=self.is_fully_complex,
                          init=init_weights)
            )
        
        return nn.ModuleList(conv_blocks)
        
    def _create_rnn_block(self):
        if self.is_fully_complex:
            return ComplexGRU(input_size=self.max_filters//2,
                            hidden_size=self.max_filters//4,
                            batch_first=True, bidirectional=True)
        else:
            return nn.GRU(input_size=self.max_filters,
                          hidden_size=self.max_filters//2,
                          batch_first=True, bidirectional=True)

    def _create_fully_connected_block(self, n_sources, fc_layer_dropout_rate):
        # TODO: Allow user to choose the number of linear layers
        
        if self.is_fully_complex:
            layer_input_size = self.max_filters//2
            if self.is_parameterized:
                layer_input_size += self.n_input_channels + 1
                # Each microphone's coordinates is encoded by a complex number,
                # plus the room dimensions

            if self.activation == "relu":
                activation = ComplexReLU
            elif self.activation == "prelu":
                activation = ComplexPReLU

            # TODO: implement dropout for complex networks
            return nn.Sequential(
                ComplexLinear(layer_input_size, layer_input_size),
                activation(),
                ComplexLinear(layer_input_size, n_sources),
            )
        else:
            layer_input_size = self.max_filters
            if self.is_parameterized:
                layer_input_size += 2*(self.n_input_channels + 1)
                # Each microphone's coordinates is encoded by two real numbers
                # plus the two room dimensions

            if self.activation == "relu":
                activation = nn.ReLU
            elif self.activation == "prelu":
                activation = nn.PReLU

            n_last_layer = 2*n_sources  # 2 cartesian dimensions for each source            
            if fc_layer_dropout_rate > 0:
                return nn.Sequential(
                    nn.Linear(layer_input_size, layer_input_size),
                    activation(),
                    nn.Dropout(fc_layer_dropout_rate),
                    nn.Linear(layer_input_size, n_last_layer),
                    nn.Dropout(fc_layer_dropout_rate)
                )
            else:
                return nn.Sequential(
                    nn.Linear(layer_input_size, layer_input_size),
                    activation(),
                    nn.Linear(layer_input_size, n_last_layer),
                )
    
    def track_feature_maps(self):
        "Make all the intermediate layers accessible through the 'feature_maps' dictionary"

        self.feature_maps = {}

        hook_fn = self._create_hook_fn("stft")
        self.feature_extractor.register_forward_hook(hook_fn)
        
        for i, conv_layer in enumerate(self.conv_blocks):
            hook_fn = self._create_hook_fn(f"conv_{i}")
            conv_layer.register_forward_hook(hook_fn)
        
        hook_fn = self._create_hook_fn("rnn")
        self.rnn.register_forward_hook(hook_fn)

        hook_fn = self._create_hook_fn("fully_connected")
        self.fully_connected.register_forward_hook(hook_fn)

    def _create_hook_fn(self, layer_id):
        def fn(_, __, output):
            if type(output) == tuple:
                output = output[0]
            self.feature_maps[layer_id] = output.detach().cpu()
        return fn


def complex_to_real(x, mode="real_imag", axis=1):
    if mode == "real_imag":
        x = torch.cat([x.real, x.imag], axis=axis)
    elif mode == "magnitude":
        x = x.abs()
    elif mode == "phase":
        x = x.angle()
    elif mode == "amp_phase":
        x = torch.cat([x.abs(), x.angle()], axis=axis)
    else:
        raise ValueError(f"Invalid complex mode :{mode}")

    return x
