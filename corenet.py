import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CvConfidence(nn.Module):
    def __init__(self, device):
        super(CvConfidence, self).__init__()
        self._device = device

    def forward(self, prob_volume, disps):
        N, _, H, W = prob_volume.shape

        # generate coordinates
        n_coords = torch.arange(N, device=prob_volume.device, dtype=torch.long)
        n_coords = n_coords.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x_coords = torch.arange(W, device=prob_volume.device, dtype=torch.long)
        y_coords = torch.arange(H, device=prob_volume.device, dtype=torch.long)
        y_coords = y_coords.unsqueeze(-1)
        # xl, yl = torch.meshgrid((x_coords, y_coords)) # with torch >= 0.4.1

        nl = n_coords.repeat((1, 1, H, W)).long()
        xl = x_coords.repeat((N, 1, H, 1)).long()
        yl = y_coords.repeat((N, 1, 1, W)).long()

        cv_confidence = prob_volume[nl, torch.round(disps).long(), yl, xl]
        return cv_confidence

class LrDistance(nn.Module):
    def __init__(self, device):
        super(LrDistance, self).__init__()
        self._device = device

    def forward(self, disps_lr, disps_rl):
        dispsLR = disps_lr.float()
        dispsRL = disps_rl.float()

        S, _, M, N = dispsLR.shape

        # generate coordinates
        x_coords = torch.arange(N, device=self._device, dtype=torch.float)
        y_coords = torch.arange(M, device=self._device, dtype=torch.float)

        xl = x_coords.repeat((S, M, 1)).unsqueeze(1)
        yl = y_coords.repeat((S, N, 1)).transpose(2, 1).unsqueeze(1)

        xr = xl - dispsLR

        # normalize coordinates for sampling between [-1, 1]
        xr_normed = (2 * xr / (N - 1)) - 1.0
        yl_normed = (2 * yl / (M - 1)) - 1.0

        # coords must have sahpe N x OH x OW x 2
        sample_coords = torch.stack((xr_normed[:, 0], yl_normed[:, 0]), dim=-1)
        dispsRL_warped = nn.functional.grid_sample(dispsRL, sample_coords)

        lr_distance = torch.abs(dispsLR + dispsRL_warped)
        lr_distance[(xr >= N) | (xr < 0)] = 100.0

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(), plt.imshow(disps_lr[0,0].detach().cpu()), plt.title('LR')
        # plt.figure(), plt.imshow(disps_rl[0,0].detach().cpu()), plt.title('RL')
        # plt.figure(), plt.imshow(dispsRL_warped[0,0].detach().cpu()), plt.title('RL WARPED')
        # plt.figure(), plt.imshow(lr_distance[0,0].detach().cpu()), plt.title('DIST')

        # import pdb
        # pdb.set_trace()

        return lr_distance

class LrCheck(nn.Module):
    def __init__(self, device, eps):
        super(LrCheck, self).__init__()
        self._device = device
        self._eps = eps

    def forward(self, lr_dists):
        lr_mask = torch.ones_like(lr_dists)
        lr_mask[lr_dists > self._eps] = 0.0

        zero = torch.tensor([0], dtype=torch.float).to(self._device)
        confidence = torch.max(self._eps - lr_dists, zero) / self._eps

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(confidence.detach()[0,0]), plt.title('conf')
        # plt.figure(), plt.imshow(lr_mask.detach()[0,0]), plt.title('mask')

        # import pdb
        # pdb.set_trace()

        return lr_mask, confidence

class TemperatureSoftmax(nn.Module):
    def __init__(self, dim, init_temp=1.0):
        nn.Module.__init__(self)
        self.T = nn.Parameter(torch.ones(1).float().cuda() * init_temp, requires_grad=True)
        self.dim = dim

    def forward(self, x):
        return F.softmax(x / self.T, dim=self.dim)

class TemperatureSoftmin(nn.Module):
    def __init__(self, dim, init_temp=1.0):
        nn.Module.__init__(self)
        self.T = nn.Parameter(torch.ones(1).float().cuda() * init_temp, requires_grad=True)
        self.dim = dim

    def forward(self, x):
        return F.softmin(x / self.T, dim=self.dim)               

class Pad(nn.Module):
    def __init__(self, divisor, extra_pad=(0, 0)):
        nn.Module.__init__(self)
        self.divisor = divisor
        self.extra_pad_h = extra_pad[0] # pad at top and bottom with specified value
        self.extra_pad_w = extra_pad[1] # pad at left and right with specified value

        self.l = 0
        self.r = 0
        self.t = 0
        self.b = 0

    def pad(self, x):
        N, C, H, W = x.shape

        w_add = 0
        while W % self.divisor != 0:
            W += 1
            w_add += 1

        h_add = 0
        while H % self.divisor != 0:
            H += 1
            h_add += 1

        # additionally pad kitti imgs
        self.l = self.extra_pad_w + np.ceil(w_add / 2.0).astype('int')
        self.r = self.extra_pad_w + np.floor(w_add / 2.0).astype('int')
        self.t = self.extra_pad_h + np.ceil(h_add / 2.0).astype('int')
        self.b = self.extra_pad_h + np.floor(h_add / 2.0).astype('int')

        padded = F.pad(x, (self.l, self.r, self.t, self.b), mode='reflect')
        return padded

    def forward(self, x):
        return self.pad(x)

class Unpad(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def unpad_NCHW(self, x, l, r, t, b):
        x = x[:, :, t:, l:]
        if b > 0:
            x = x[:, :, :-b, :]
        if r > 0:
            x = x[:, :, :, :-r]
        return x.contiguous()

    def unpad_NHWC(self, x, l, r, t, b):
        x = x[:, t:, l:, :]
        if b > 0:
            x = x[:, :-b, :, :]
        if r > 0:
            x = x[:, :, :-r, :]
        return x.contiguous()


    def unpad_flow_N2HWC(self, x, l, r, t, b):
        x = x[:, :, t:, l:, :]
        if b > 0:
            x = x[:, :,  :-b, :, :]
        if r > 0:
            x = x[:, :, :, :-r, :]
        return x.contiguous()

    def unpad_flow_N2CHW(self, x, l, r, t, b):
        x = x[:, :, :, t:, l:]
        if b > 0:
            x = x[:, :, :, :-b, :]
        if r > 0:
            x = x[:, :, :, :, :-r]
        return x.contiguous()

    def forward(self, x, l, r, t, b, NCHW=True):
        if NCHW:
            if len(x.shape) == 4:
                return self.unpad_NCHW(x, l, r, t, b)
            else:
                return self.unpad_flow_N2CHW(x, l, r, t, b)
        else:
            if len(x.shape) == 4:
                return self.unpad_NHWC(x, l, r, t, b)
            else:
                return self.unpad_flow_N2HWC(x, l, r, t, b)

class PadUnpad(Pad, Unpad):
    def __init__(self, net, divisor=1, extra_pad=(0, 0)):
        Pad.__init__(self, divisor, extra_pad)
        Unpad.__init__(self)

        self.net = net

    def forward(self, ipt):
        out = self.net.forward(self.pad(ipt))
        res = []
        for o in out:
            res.append(self.unpad_NCHW(o, self.l, self.r, self.t, self.b))
        return res

class PaddedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super(PaddedConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, dilation=dilation, bias=bias)

        self.dilation = dilation

    def forward(self, x):
        pd = (self.conv.kernel_size[0] // 2) * self.dilation
        x = F.pad(x, (pd, pd, pd, pd), mode='reflect')
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0,
                 activation='ReLU', transposed=False, with_bn=True, leaky_alpha=1e-2):
        super(ResidualBlock, self).__init__()

        self.convbnact1 = ConvBatchNormAct(in_channels, out_channels, kernel_size, stride, dilation,
                                      padding, activation, transposed, with_bn)

        bias = False # because the parameter is already contained in group-norm!!
        self.conv2 = PaddedConv2d(out_channels, out_channels, kernel_size, stride, dilation, bias)
        num_groups = out_channels
        self.bn2 = nn.GroupNorm(num_groups, out_channels, affine=True)

        if activation.lower() == 'relu':
            self.act2 = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            self.act2 = nn.LeakyReLU(negative_slope=leaky_alpha, inplace=True)
        elif activation.lower() == 'elu':
            self.act2 = nn.ELU(inplace=True)
        else:
            raise NotImplementedError("Activation " + activation + " is currently not implemented!")

    def forward(self, x):
        residual = x
        x = self.convbnact1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.act2(x)
        return x



class ConvBatchNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0,
                 activation='ReLU', transposed=False, with_bn=True):
        super(ConvBatchNormAct, self).__init__()

        if with_bn:
            bias = False
        else:
            bias = True

        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        else:
            self.conv = PaddedConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                     dilation=dilation, bias=bias)

        self.bn = None
        if with_bn:
            #self.bn = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
            num_groups = out_channels
            self.bn = nn.GroupNorm(num_groups, out_channels, affine=True)


        if activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=True)
        elif activation.lower() == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            raise NotImplementedError("Activation " + activation + " is currently not implemented!")

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self._divisor = -1
        self._out_channels = -1

    @property
    def divisor(self):
        return self._divisor

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def num_output_levels(self):
        return self._num_output_levels

    def forward(self, *input):
        raise NotImplementedError


class StereoUnaryUnetDyn(BaseNet):
    def __init__(self, multi_level_output=False, activation='relu',
                 with_bn=True, with_upconv=False, with_output_bn=True):
        super(StereoUnaryUnetDyn, self).__init__()

        self.multi_level_output = multi_level_output
        self.with_upconv = with_upconv
        self.with_output_bn = with_output_bn
        self._divisor = 8
        self._out_channels = [64]
        if multi_level_output:
            self._out_channels = [64, 64, 64]

        self._num_output_levels = 1
        if multi_level_output:
            self._num_output_levels = 3

        self._out_channels = [32]
        if multi_level_output:
            self._out_channels = [32, 32, 32]

        self.conv1 = ConvBatchNormAct(3, 16, 3, padding=1, activation=activation, with_bn=with_bn)
        self.conv2 = ConvBatchNormAct(16, 16, 3, padding=1, activation=activation, with_bn=with_bn)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = ConvBatchNormAct(16, 32, 3, padding=1, activation=activation, with_bn=with_bn)
        self.conv4 = ConvBatchNormAct(32, 32, 3, padding=1, activation=activation, with_bn=with_bn)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = ConvBatchNormAct(32, 64, 3, padding=1, activation=activation, with_bn=with_bn)
        self.conv6 = ConvBatchNormAct(64, 64, 3, padding=1, activation=activation, with_bn=with_bn)

        if self.with_upconv:
            self.upconv7 = ConvBatchNormAct(64, 32, 3, stride=2, padding=0, activation=activation,
                                            transposed=True, with_bn=with_bn)
            self.conv8 = ConvBatchNormAct(64, 32, 3, padding=1, activation=activation,
                                          with_bn=with_bn)
        else:
            self.conv8 = ConvBatchNormAct(96, 32, 3, padding=1, activation=activation,
                                          with_bn=with_bn)
        self.conv9 = ConvBatchNormAct(32, 32, 3, padding=1, activation=activation, with_bn=with_bn)

        if self.with_upconv:
            self.upconv10 = ConvBatchNormAct(32, 16, 3, stride=2, padding=0, activation=activation,
                                             transposed=True, with_bn=with_bn)
            self.conv11 = ConvBatchNormAct(32, 32, 3, padding=1, activation=activation,
                                           with_bn=with_bn)
        else:
            self.conv11 = ConvBatchNormAct(48, 32, 3, padding=1, activation=activation,
                                           with_bn=with_bn)

        self.conv12 = PaddedConv2d(32, 32, 3)
        self.bn12 = None
        if with_bn and with_output_bn:
#            self.bn12 = nn.BatchNorm2d(32, affine=True, track_running_stats=False)
            self.bn12 = nn.GroupNorm(32, 32, affine=True)

        if self.multi_level_output:
            self.conv_lvl1 = PaddedConv2d(32, 32, 3)
            self.conv_lvl2 = PaddedConv2d(64, 32, 3)

    def forward(self, x_in):
        x = x_in
        x = self.conv1(x)
        lvl0 = self.conv2(x)
        x = self.pool1(lvl0)

        x = self.conv3(x)
        lvl1 = self.conv4(x)
        x = self.pool2(lvl1)

        x = self.conv5(x)
        x = self.conv6(x)

        if self.multi_level_output:
            lvl2_out = self.conv_lvl2(x)

        if self.with_upconv:
            x = self.upconv7(x)[:, :, 1:, 1:]
        else:
            x = F.interpolate(x, lvl1.shape[2:], mode='bilinear')
        x = torch.cat([lvl1, x], dim=1)

        x = self.conv8(x)
        x = self.conv9(x)

        if self.multi_level_output:
            lvl1_out = self.conv_lvl1(x)

        if self.with_upconv:
            x = self.upconv10(x)[:, :, :-1, :-1]
        else:
            x = F.interpolate(x, lvl0.shape[2:], mode='bilinear')
        x = torch.cat([lvl0, x], dim=1)

        x = self.conv11(x)
        x = self.conv12(x)
        if self.bn12:
            x = self.bn12(x)

        if self.multi_level_output:
            return x, lvl1_out, lvl2_out

        return [x]