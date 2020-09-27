import torch
import torch.nn.functional as F
import torch.distributed as distributed

import math

from torch import nn
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups, dilation = dilation)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=True, dilation = 1, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False, dilation = dilation)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

# LSFE module
class LSFE(nn.Module):
    def __init__(self, ):
        super(LSFE, self).__init__()
        self.conv1 = SeparableConvBlock(88, 256)
        self.conv2 = SeparableConvBlock(256, 256)
        self.conv3 = SeparableConvBlock(256, 256)
        self.conv4 = SeparableConvBlock(256, 256)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        return x

# Mismatch Correction Module (MC)
class CorrectionModule(nn.Module):
    def __init__(self):
        super(CorrectionModule, self).__init__()
        self.conv1 = SeparableConvBlock(256, 256)
        self.conv2 = SeparableConvBlock(256, 256)
        self.conv3 = SeparableConvBlock(256, 256)
        self.conv4 = SeparableConvBlock(256, 256)
        self.conv5 = SeparableConvBlock(256, 256)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        ## upsampling 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.up(x)
        return x

# Dense Prediction Cells (DPC)
class DPC(nn.Module):
    def __init__(self, height, width, channels = 256):
        super(DPC, self).__init__()

        self.height = height
        self.width = width

        self.conv1 = SeparableConvBlock(88, 256, 3, dilation=(1, 6))
        self.up1 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.conv2 = SeparableConvBlock(256, 256, 3, dilation=(1, 1))
        self.up2 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.conv3 = SeparableConvBlock(256, 256, 3, dilation=(6, 12))
        self.up3 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.up_tocalculate18x3 = nn.Upsample((36, 64), mode='bilinear')
        self.conv4 = SeparableConvBlock(256, 256, 3, dilation=(15, 12))
        self.up4 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.conv5 = SeparableConvBlock(256, 256, 3, dilation=(6,3))
        self.up5 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.lastconv = nn.Conv2d(1280, 256, 1)

    def forward(self, x):

        x = self.conv1(x)
        x1 = self.up1(x)
        
        x2 = self.conv2(x1)
        x2 = self.up2(x2)  

        x3 = self.conv3(x1)
        x3 = self.up3(x3)

        x4 = x1
        if( self.height < 33 ):
          x4 = self.up_tocalculate18x3(x4)
        x4 = self.conv4(x4)
        x4 = self.up4(x4)    

        x5 = self.conv5(x4)
        x5 = self.up5(x5)

        cat = torch.cat(( x1, x2, x3, x4, x5), dim = 1)

        cat = self.lastconv(cat)        

        return cat

class SemanticHead(nn.Module):
    def __init__(self):
        super(SemanticHead, self).__init__()
        self.dpcp32 = DPC(24, 24)
        self.dpcp16 = DPC(48, 48)
        self.lsfep8 = LSFE()

        self.up_p32_1 = nn.Upsample((48, 48), mode='bilinear')
        self.up_p32_2 = nn.Upsample((96, 96), mode='bilinear')
        self.up_p32_3 = nn.Upsample((256, 256), mode='bilinear')

        self.conv1_32_1 = SeparableConvBlock(256, 256, 3)
        self.conv1_32_2 = SeparableConvBlock(256, 256, 3)
        self.conv1_32_3 = SeparableConvBlock(256, 256, 3)
        self.conv1_32_4 = SeparableConvBlock(256, 256, 3)
        
        self.conv2_32_1 = SeparableConvBlock(256, 256, 3)
        self.conv2_32_2 = SeparableConvBlock(256, 256, 3)
        self.conv2_32_3 = SeparableConvBlock(256, 256, 3)
        self.conv2_32_4 = SeparableConvBlock(256, 256, 3)

        self.conv1_8 = SeparableConvBlock(256, 256, 3)
        self.conv2_8 = SeparableConvBlock(256, 256, 3)
        self.conv3_8 = SeparableConvBlock(256, 256, 3)
        self.conv4_8 = SeparableConvBlock(256, 256, 3)
        
        self.conv1_16_1 = SeparableConvBlock(256, 256, 3)
        self.conv1_16_2 = SeparableConvBlock(256, 256, 3)
        self.conv1_16_3 = SeparableConvBlock(256, 256, 3)
        self.conv1_16_4 = SeparableConvBlock(256, 256, 3)

        self.conv2_16_1 = SeparableConvBlock(256, 256, 3)
        self.conv2_16_2 = SeparableConvBlock(256, 256, 3)
        self.conv2_16_3 = SeparableConvBlock(256, 256, 3)
        self.conv2_16_4 = SeparableConvBlock(256, 256, 3)

        self.mc1 = CorrectionModule()

        self.up_p16_1 = nn.Upsample((96, 96), mode='bilinear')
        self.up_p16_2 = nn.Upsample((256, 256), mode='bilinear')

        self.up_p8_1 = nn.Upsample((256, 256), mode='bilinear')

        self.last1_1 = nn.Conv2d(768, 256, 1)
        self.last1_2 = nn.Conv2d(256, 256, 3)
        self.last1_3 = nn.Conv2d(256, 256, 3)
        self.last1_4 = nn.Conv2d(256, 256, 3)
        self.last2 = nn.Conv2d(256, 20, 1) ####### NEED TO CHANGE OUTPUT CHANNELS
        self.uplast1 = nn.Upsample((512, 512), mode = 'bilinear')
        self.uplast2 = nn.Upsample((512, 1024), mode = 'bilinear')
    
    
    def forward(self, p32, p16, p8):

        d32 = self.dpcp32(p32)
        lp8 = self.lsfep8(p8)

        d16 = self.dpcp16(p16)
        up32 = self.up_p32_1(d32)
        
        #to calculate p8
        add1 = torch.add(up32, d16)
        up16 = self.mc1(add1)
        add2 = torch.add(up16, lp8)
        conv_8 = self.conv1_8(add2)
        conv_8 = self.conv2_8(conv_8)
        conv_8 = self.conv3_8(conv_8)
        conv_8 = self.conv4_8(conv_8)
        cat3 = self.up_p8_1(conv_8)

        #to calculate p32
        conv1_32_ = self.conv1_32_1(up32)
        conv1_32_ = self.conv1_32_2(conv1_32_)
        conv1_32_ = self.conv1_32_3(conv1_32_)
        conv1_32_ = self.conv1_32_4(conv1_32_)

        conv1_32_ = self.up_p32_2(conv1_32_)
        conv2_32_ = self.conv2_32_1(conv1_32_)
        conv2_32_ = self.conv2_32_2(conv2_32_)
        conv2_32_ = self.conv2_32_3(conv2_32_)
        conv2_32_ = self.conv2_32_4(conv2_32_)
        cat1 = self.up_p32_3(conv2_32_)

        #to calculate p16
        conv1_16_ = self.conv1_16_1(d16)
        conv1_16_ = self.conv1_16_2(conv1_16_)
        conv1_16_ = self.conv1_16_3(conv1_16_)
        conv1_16_ = self.conv1_16_4(conv1_16_)
        conv1_16_ = self.up_p16_1(conv1_16_)
        
        conv2_16_ = self.conv2_16_1(conv1_16_)
        conv2_16_ = self.conv2_16_2(conv2_16_)
        conv2_16_ = self.conv2_16_3(conv2_16_)
        conv2_16_ = self.conv2_16_4(conv2_16_)
        cat2 = self.up_p16_2(conv2_16_)

        cat = torch.cat(( cat1, cat2, cat3), dim = 1)

        cat = self.last1_1(cat)
        cat = self.last1_2(cat)
        cat = self.last1_3(cat)
        cat = self.last1_4(cat)
        cat = self.uplast1(cat)
        cat = self.last2(cat)
        cat = self.uplast2(cat)
        
        return cat