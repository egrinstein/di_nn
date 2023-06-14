import torch
import torch.nn as nn

from di_nn.utils.conv_block import (
    DEFAULT_CONV_CONFIG, ConvBlock
)
from di_nn.utils.model_utilities import init_gru, init_layer


# Base generic class to be inherited by the SSL specific network
class DICRNN(nn.Module):
    def __init__(self, n_input_channels, n_output,
                 pool_type="avg", pool_size=(1,2), kernel_size=(2, 2),
                 conv_layers_config=DEFAULT_CONV_CONFIG,
                 fc_layer_dropout_rate=0.5,
                 activation="relu",
                 init_layers=True,
                 n_metadata=0,
                 is_early_fusion=False,
                 use_embedding_metadata_layer=False,
                 **kwargs):
        
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.n_metadata_unaware_features = conv_layers_config[-1]["n_channels"]
        self.is_metadata_aware = n_metadata > 0 # Parameterized Neural Network:
                                                   # concatenate the microphone coordinates to the features before
                                                   # feeding them to the fully connected layers
        self.is_early_fusion = is_early_fusion
        self.use_embedding_metadata_layer = use_embedding_metadata_layer

        if self.is_early_fusion:
            self.n_input_channels *= 2 # Add one extra channel per input for the metadata

        # 2. Create feature extraction network
        self.feature_extraction_network = FeatureExtractionNetwork(
                                            self.n_input_channels,
                                            conv_layers_config,
                                            init_layers,
                                            pool_type,
                                            pool_size,
                                            kernel_size,
                                            activation)    
    
        # 3. Create metadata fusion network
        n_input_metadata_fusion_network = self.n_metadata_unaware_features
        if self.is_metadata_aware and not is_early_fusion:
            n_input_metadata_fusion_network += n_metadata

        if use_embedding_metadata_layer:
            self.metadata_embedding_layer = nn.Linear(n_metadata, n_metadata)

        self.metadata_fusion_network = MetadataFusionNetwork(
                                            n_input_metadata_fusion_network,
                                            n_output,
                                            dropout_rate=fc_layer_dropout_rate,
                                            activation=activation,
                                            init_layers=init_layers)
    
    def forward(self, x):
        if self.is_metadata_aware:
            metadata = x["metadata"]
            x = x["signal"]

            if self.is_early_fusion:
                # Concatenate metadata before sending to Feature extraction network
                x = torch.cat([x, metadata], dim=1)
        # (batch_size, num_channels, time_steps, num_features)
        # In our case, "num_features" refers to the number of Fourier coefficients

        x = self.feature_extraction_network(x)

        if self.is_metadata_aware and not self.is_early_fusion:
            # Concatenate metadata before sending to fully connected layer,
            # if late fusion
            if self.use_embedding_metadata_layer:
                metadata = self.metadata_embedding_layer(metadata)

            x = torch.cat([x, metadata], dim=1)
            # (batch_size, n_metadata_unaware_features + n_metadata)
        
        # 5. Fully connected layer
        x = self.metadata_fusion_network(x)
        # (batch_size, class_num)

        return x


class MetadataFusionNetwork(nn.Module):
    def __init__(self, n_input, n_output, dropout_rate=0, activation="relu", init_layers=True):
        super().__init__()
        
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "prelu":
            activation = nn.PReLU

        if dropout_rate > 0:
            self.fc_network = nn.Sequential(
                nn.Linear(n_input, n_input),
                activation(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_input, n_output),
                nn.Dropout(dropout_rate)
            )
        else:
            self.fc_network = nn.Sequential(
                nn.Linear(n_input, n_input),
                activation(),
                nn.Linear(n_input, n_output),
            )
        
        if init_layers:
            init_layer(self.fc_network)

    def forward(self, x):
        return self.fc_network(x)


class FeatureExtractionNetwork(nn.Module):
    def __init__(self, n_input_channels,
                       conv_layers_config,
                       init_layers,
                       pool_type,
                       pool_size,
                       kernel_size,
                       activation):
    
        super().__init__()

        # 1. Store configuration
        self.n_input_channels = n_input_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.max_filters = conv_layers_config[-1]["n_channels"]

        # 2. Create convolutional blocks
        self.conv_blocks = self._create_conv_blocks(conv_layers_config, init_weights=init_layers)

        # 3. Create recurrent block
        self.rnn = self._create_rnn_block()

        if init_layers:
            init_gru(self.rnn)

    def forward(self, x):
        # (batch_size, num_channels, time_steps, num_features)
        # In our case, "num_features" refers to the number of Fourier coefficients

        # 1. Extract features using convolutional layers
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # (batch_size, feature_maps, time_steps', num_features')

        # 2. Average across all input features
        x = torch.mean(x, dim=3)
        # (batch_size, feature_maps, time_steps)

        # Preprocessing for RNN
        x = x.transpose(1,2)
        # (batch_size, time_steps, feature_maps):
        
        # 3. Use features as input to RNN
        (x, _) = self.rnn(x)
        # (batch_size, time_steps, feature_maps):
        # Average across all time steps
        x = torch.mean(x, dim=1)
        # (batch_size, feature_maps)

        return x

    def _create_conv_blocks(self, conv_layers_config, init_weights):
        
        conv_blocks = [
            ConvBlock(self.n_input_channels, conv_layers_config[0]["n_channels"],
                      block_type=conv_layers_config[0]["type"],
                      dropout_rate=conv_layers_config[0]["dropout_rate"],
                      pool_size=self.pool_size,
                      activation=self.activation,
                      kernel_size=self.kernel_size,
                      init=init_weights)
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
                          init=init_weights)
            )
        
        return nn.ModuleList(conv_blocks)
        
    def _create_rnn_block(self):
        return nn.GRU(input_size=self.max_filters,
                      hidden_size=self.max_filters//2,
                      batch_first=True, bidirectional=True)
