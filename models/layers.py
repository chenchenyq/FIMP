from typing import List, Optional
from torch import Tensor, reshape, stack
import torch
import numpy as np
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Module,
    PReLU,
    Sequential,
    Sigmoid,
)


class FourierUnit(Module):
    def __init__(self, in_channel, out_channel, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = Sequential(
            Conv2d(in_channel * 2, out_channel * 2, 1, 1, 0, groups=self.groups, bias=False),
            BatchNorm2d(out_channel * 2),
            PReLU(),
        )
        self.conv7x7 = Conv2d(2, 1, 7, padding=7//2)
        self.sigmoid = Sigmoid()

    def forward(self, x, y):
        batch, c, h, w = x.size()
        # (batch, c, h, w, 2)
        ffted_x = torch.fft.fft2(x, dim=(-2, -1))
        # ffted_x = torch.fft.fftshift(ffted_x, dim=(-2, -1))
        ffted_y = torch.fft.fft2(y, dim=(-2, -1))
        # ffted_y = torch.fft.fftshift(ffted_y, dim=(-2, -1))
        fftedx = torch.stack((ffted_x.real, ffted_x.imag), -1)
        fftedy = torch.stack((ffted_y.real, ffted_y.imag), -1)
        # (batch, c, 2, h, w)
        fftedx = fftedx.permute(0, 1, 4, 2, 3).contiguous()
        fftedy = fftedy.permute(0, 1, 4, 2, 3).contiguous()
        fftedx = fftedx.view((batch, -1,) + fftedx.size()[3:])
        fftedy = fftedy.view((batch, -1,) + fftedy.size()[3:])
        # (batch, c*2, h, w/2+1)
        ffted = fftedx - fftedy
        ffted = self.conv_layer(ffted)
        max_out, _ = torch.max(ffted, dim=1, keepdim=True)
        avg_out = torch.mean(ffted, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv7x7(torch.cat([max_out, avg_out], dim=1)))

        fftedx = fftedx * mask
        fftedy = fftedy * mask
        fftedx = fftedx.view((batch, -1, 2,) + fftedx.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        # fftedx = torch.fft.ifftshift(fftedx, dim=(2, 3))
        fftedy = fftedy.view((batch, -1, 2,) + fftedy.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        # fftedy = torch.fft.ifftshift(fftedy, dim=(2, 3))
        ifftedx = torch.fft.ifft2(torch.complex(fftedx[..., 0], fftedx[..., 1]), dim=(-2, -1))
        ifftedy = torch.fft.ifft2(torch.complex(fftedy[..., 0], fftedy[..., 1]), dim=(-2, -1))
        output_x = ifftedx.real
        output_y = ifftedy.real
        return output_x, output_y


class TemporalInteractionBlock(Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mix = MixingBlock(in_channel)
        self.conv_sub = Sequential(
            Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_channel),
            PReLU()
        )
        self.conv_cat = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channel),
            PReLU()
        )

    def forward(self, x1, x2):
        x_sub = self.conv_sub(torch.abs(x1 - x2))
        x_mix = self.mix(x1, x2)
        x_fus = x_mix + x_mix.mul(x_sub)
        x = self.conv_cat(x_fus)
        return x


class MixingBlock(Module):
    def __init__(self, in_channel):
        super().__init__()
        self.convmix = Sequential(
            Conv2d(in_channel*2, in_channel, 3, groups=in_channel, padding=1),
            BatchNorm2d(in_channel),
            PReLU(),
        )

    def forward(self, x, y):
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self.convmix(mixed)


class Up(Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        self.convolution1x1 = Sequential(
            Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1),
            BatchNorm2d(in_channel),
            PReLU(),
        )
        self.convolution = Sequential(
            Conv2d(out_channel, out_channel, 3, 1, padding=1),
            BatchNorm2d(out_channel),
            PReLU(),
            Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            BatchNorm2d(out_channel),
            PReLU(),
        )

    def forward(self, x, y: Optional[Tensor] = None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
            x = self.convolution1x1(x)
            x = self.up(x)
            x = self.convolution(x)
        else:
            x = self.up(x)
            x = self.convolution(x)
        return x


class Classifier(Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.linear = Sequential(
            Conv2d(in_channel, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class Concat(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in*2, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class Add(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = x1 + x2
        x = self.conv(x)
        return x


class Sub(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = Sequential(
            Conv2d(self.ch_in, self.ch_out, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.ch_out),
            PReLU()
        )

    def forward(self, x1, x2):
        x = x1 - x2
        x = self.conv(x)
        return x


class FourierUnit3(Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit3, self).__init__()
        self.groups = groups
        self.conv_layer = Sequential(
            Conv2d(in_channels * 2, out_channels * 2, 1, 1, 0, groups=self.groups),
            BatchNorm2d(out_channels * 2),
            PReLU(),
        )

        self.conv1 = Conv2d(2, 1, 7, padding=7//2)
        self.sigmoid = Sigmoid()
        # self.relu = ReLU(inplace=True)


    def forward(self, x, y):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted_x = torch.fft.fft2(x, dim=(-2, -1))
        ffted_y = torch.fft.fft2(y, dim=(-2, -1))
        amp_x, pha_x = torch.abs(ffted_x), torch.angle(ffted_x)
        amp_y, pha_y = torch.abs(ffted_y), torch.angle(ffted_y)
        fftedx = torch.stack((amp_x, pha_x), -1)
        fftedy = torch.stack((amp_y, pha_y), -1)
        fftedx = fftedx.permute(0, 1, 4, 2, 3).contiguous()
        fftedy = fftedy.permute(0, 1, 4, 2, 3).contiguous()
        fftedx = fftedx.view((batch, -1,) + fftedx.size()[3:])
        fftedy = fftedy.view((batch, -1,) + fftedy.size()[3:])
        # (batch, c, 2, h, w/2+1)

        # a = torch.sub(fftedx, fftedy)
        ffted = fftedx - fftedy

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        max_out, _ = torch.max(ffted, dim=1, keepdim=True)
        avg_out = torch.mean(ffted, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv1(torch.cat([max_out, avg_out], dim=1)))

        fftedx = fftedx * mask
        fftedy = fftedy * mask
        fftedx = fftedx.view((batch, -1, 2,) + fftedx.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        fftedy = fftedy.view((batch, -1, 2,) + fftedy.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        ifftedx = torch.fft.ifft2(fftedx[..., 0] * torch.exp(1j * fftedx[..., 1]), dim=(-2, -1))
        ifftedy = torch.fft.ifft2(fftedy[..., 0] * torch.exp(1j * fftedy[..., 1]), dim=(-2, -1))

        # ifftedx = torch.fft.ifft2(torch.complex(fftedx[..., 0], fftedx[..., 1]), dim=(-2, -1))
        # ifftedy = torch.fft.ifft2(torch.complex(fftedy[..., 0], fftedy[..., 1]), dim=(-2, -1))
        # ifftedx = torch.fft.ifft2(torch.complex(f1[..., 0], f1[..., 1]), dim=(-2, -1))
        # ifftedy = torch.fft.ifft2(torch.complex(f2[..., 0], f2[..., 1]), dim=(-2, -1))
        output_x = ifftedx.real
        output_y = ifftedy.real
        return output_x, output_y