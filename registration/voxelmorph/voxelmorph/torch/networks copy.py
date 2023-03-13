import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def default_unet_features():
    net_channels = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return net_channels

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 shape=None,
                 in_channels=None,
                 net_channels=None,
                 net_depth=None,
                 pool_kernel_size=2,
                 channel_mult=1,
                 conv_per_layer=1,
                 half_res=False):
        """
        Parameters:
            shape: Input shape. e.g. (192, 192, 192)
            in_channels: Number of input features.
            net_channels: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            net_depth: Number of levels in unet. Only used when net_channels is an integer. 
                Default is None.
            pool_kernel_size: Maxpool layer kernel size
            channel_mult: Per-level feature multiplier. Only used when net_channels is an integer. 
                Default is 1.
            conv_per_layer: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(shape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if net_channels is None:
            net_channels = default_unet_features()

        # build feature list automatically
        if isinstance(net_channels, int):
            if net_depth is None:
                raise ValueError('must provide unet net_depth if net_channels is an integer')
            feats = np.round(net_channels * channel_mult ** np.arange(net_depth)).astype(int)
            net_channels = [
                np.repeat(feats[:-1], conv_per_layer),
                np.repeat(np.flip(feats), conv_per_layer)
            ]
        elif net_depth is not None:
            raise ValueError('cannot use net_depth if net_channels is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        encoder_channels, decoder_channels = net_channels
        pivot = len(encoder_channels)
        final_convs = decoder_channels[pivot:]
        decoder_channels = decoder_channels[:pivot]
        self.net_depth = int(pivot / conv_per_layer) + 1

        if isinstance(pool_kernel_size, int):
            pool_kernel_size = [pool_kernel_size] * self.net_depth

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in pool_kernel_size]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in pool_kernel_size]

        # configure encoder (down-sampling path)
        prev_nf = in_channels
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.net_depth - 1):
            convs = nn.ModuleList()
            for conv in range(conv_per_layer):
                nf = encoder_channels[level * conv_per_layer + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.net_depth - 1):
            convs = nn.ModuleList()
            for conv in range(conv_per_layer):
                nf = decoder_channels[level * conv_per_layer + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.net_depth - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            #print(self.pooling[level])
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.net_depth - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
