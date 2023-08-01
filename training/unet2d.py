'''
Codes are from:
https://github.com/jaxony/unet-pytorch/blob/master/model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from torch_utils import misc
from training.networks import Conv2DNorm, DownsampleBlock, UpsampleBlock, Conv2dLayer, SynthesisBlock

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv1x1(self.in_channels, self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        return x


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class DownConvNorm(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConvNorm, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = Conv2DNorm(self.in_channels, self.out_channels)
        self.conv2 = Conv2DNorm(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class DownConvNormMod(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, **synthesis_kwargs):
        super(DownConvNormMod, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv = SynthesisBlock(
            self.in_channels, self.out_channels, up=1, is_last=False,
            **synthesis_kwargs
            )
        self.num_ws = self.conv.num_conv + self.conv.num_torgb
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, ws, img=None, **synthesis_kwargs):
        x, img = self.conv(x, img, ws, up_sample=False, **synthesis_kwargs)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        # self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(self.conv2(x))
        return x


class UNet2D(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels, out_channels,
                 f_maps=64, num_levels=5, up_mode='transpose', 
                 res_rate=1,
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
            num_levels: int, number of MaxPools in the U-Net.
            f_maps: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet2D, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = out_channels
        self.in_channels = in_channels
        self.start_filts = f_maps
        self.depth = num_levels

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(num_levels):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < num_levels-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        num_ups = num_levels - 1 - int(np.log2(res_rate))
        for i in range(num_ups):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

class UNet2DEncoder(nn.Module):
    def __init__(self, in_channels,
                 f_maps=64, num_levels=5, **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
            num_levels: int, number of MaxPools in the U-Net.
            f_maps: int, number of convolutional filters for the 
                first conv.
        """
        super(UNet2DEncoder, self).__init__()
        self.in_channels = in_channels
        self.start_filts = f_maps
        self.depth = num_levels

        self.down_convs = []
        self.up_convs = []
        self.num_out_channels = []

        # create the encoder pathway and add to a list
        for i in range(num_levels):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < num_levels-1 else False

            # down_conv = DownConv(ins, outs, pooling=pooling)
            down_conv = DownConvNorm(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
            self.num_out_channels.append(outs)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        return encoder_outs

class UNet2DEncoderMod(nn.Module):
    def __init__(self, in_channels,
                 f_maps=64, num_levels=5, init_resolution=128,
                 channel_base=None, channel_max=None, num_fp16_res=None,**synthesis_kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
            num_levels: int, number of MaxPools in the U-Net.
            f_maps: int, number of convolutional filters for the 
                first conv.
        """
        super(UNet2DEncoderMod, self).__init__()
        self.in_channels = in_channels
        self.start_filts = f_maps
        self.depth = num_levels

        self.down_convs = []
        self.up_convs = []
        self.num_out_channels = []
        self.w_dim = synthesis_kwargs["w_dim"]
        self.num_ws = 0

        # create the encoder pathway and add to a list
        for i in range(num_levels):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < num_levels-1 else False

            # down_conv = DownConv(ins, outs, pooling=pooling)
            down_conv = DownConvNormMod(
                ins, outs, pooling=pooling,
                resolution=init_resolution//(2**i),
                **synthesis_kwargs)
            self.down_convs.append(down_conv)
            self.num_out_channels.append(outs)
            self.num_ws += down_conv.num_ws

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, ws, **synthesis_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for block in self.down_convs:
                block_ws.append(ws.narrow(1, w_idx, block.num_ws))
                w_idx += block.num_ws
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x, block_ws[i], **synthesis_kwargs)
            encoder_outs.append(before_pool)
        return encoder_outs

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv3x3(self.in_channels, self.out_channels)
        self.bn = nn.SyncBatchNorm(self.out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class ConvBlockNorm(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, inter_channels=None):
        super(ConvBlockNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if inter_channels is None:
            inter_channels = self.out_channels
        self.inter_channels = inter_channels
        self.conv = Conv2DNorm(self.in_channels, self.inter_channels)
        if num_layers > 1:
            self.extra_convs = [Conv2DNorm(self.inter_channels, self.inter_channels) for _ in range(num_layers - 2)]
            self.extra_convs.append(Conv2DNorm(self.inter_channels, self.out_channels))
            self.extra_convs = nn.Sequential(*self.extra_convs)
        else:
            self.extra_convs = None

    def forward(self, x):
        x = self.conv(x)
        if self.extra_convs is not None:
            x = self.extra_convs(x)
        return x

class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % in_channels == 0
        num_stages = int(np.log2(self.out_channels/self.in_channels))
        self.conv_blocks = nn.ModuleList()
        for i in range(num_stages):
            self.conv_blocks.append(ConvBlock(in_channels * 2**i, in_channels * 2**(1 + i)))
    
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x

from torch_utils import persistence

@persistence.persistent_class
class UNet2DMod(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels, out_channels,
                 f_maps=64, num_levels=5, up_mode='transpose', 
                 res_rate=1,
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
            num_levels: int, number of MaxPools in the U-Net.
            f_maps: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        # super(UNet2DMod, self).__init__()
        super().__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = out_channels
        self.in_channels = in_channels
        self.start_filts = f_maps
        self.depth = num_levels

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        self.conv_first = Conv2dLayer(self.in_channels, self.start_filts, kernel_size=1)
        outs = self.start_filts
        for i in range(num_levels):
            ins = outs
            outs = self.start_filts*(2**i)
            pooling = True if i < num_levels-1 else False

            down_conv = DownsampleBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        num_ups = num_levels - 1 - int(np.log2(res_rate))
        for i in range(num_ups):
            ins = outs
            outs = ins // 2
            up_conv = UpsampleBlock(ins, outs, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = Conv2dLayer(outs, self.num_classes, kernel_size=1)

    def forward(self, x):
        encoder_outs = []
        x = self.conv_first(x)
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x