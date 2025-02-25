import torch.nn as nn

from di_nn.utils.model_utilities import init_layer

DEFAULT_CONV_CONFIG = [
    {"type": "single", "n_channels": 64, "dropout_rate":0},
    {"type": "single", "n_channels": 64, "dropout_rate":0},
    {"type": "single", "n_channels": 64, "dropout_rate":0},
    {"type": "single", "n_channels": 64, "dropout_rate":0},
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1),
                padding=(1,1), pool_size=(2, 2),
                block_type="double",
                init=False,
                dropout_rate=0.1,
                activation="relu"):
        
        super().__init__()
        self.block_type = block_type
        self.pool_size=pool_size
        self.dropout_rate = dropout_rate

        conv_block = nn.Conv2d
        bn_block = nn.BatchNorm2d
        dropout_block = nn.Dropout
        self.activation = nn.ReLU()
        self.pooling = nn.AvgPool2d(pool_size)

        self.conv1 = conv_block(in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=False)
        self.bn1 = bn_block(out_channels)
        self.dropout = dropout_block(dropout_rate)

        if block_type == "double": 
            self.conv2 = conv_block(in_channels=out_channels, 
                                out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=False)           
            self.bn2 = bn_block(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if init:
            self._init_weights()
        
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        if self.block_type == "double":
            x = self.activation(self.bn2(self.conv2(x)))
        x = self.pooling(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def _init_weights(self):
        init_layer(self.conv1)
        init_layer(self.bn1)
        if self.block_type == "double":
            init_layer(self.conv2)
            init_layer(self.bn2)