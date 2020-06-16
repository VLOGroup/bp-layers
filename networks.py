import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path as osp

from corenet import StereoUnaryUnetDyn, PadUnpad, PaddedConv2d, ResidualBlock

class SubNetwork(nn.Module):
    def __init__(self, args, device):
        super(SubNetwork, self).__init__()
        self.net = None
        self.pad_unpad = None
        self.args = args
        self.device = device

    def forward(self, ipt):
        return self.pad_unpad.forward(ipt)

    def freeze_parameters(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def parameter_list(self, requires_grad=None):
        def check(var):
            if requires_grad is None:
                return True
            elif requires_grad is True and var is True:
                return True
            elif requires_grad is False and var is False:
                return True
            else: return False

        params = []
        for p in self.parameters():
            if check(p.requires_grad):
                params.append(p)
        return params

    def load_parameters(self, path, device=None):
        if path is not None:
            if not osp.exists(path):
                raise FileNotFoundError("Specified unary checkpoint file does not exist!" + path)

            if device is not None:
                checkpoint = torch.load(path, map_location=device)
            else:
                checkpoint = torch.load(path)

            checkpoint = self.hook_adjust_checkpoint(checkpoint)

            self.load_state_dict(checkpoint)
            print('Successfully loaded checkpoint %s' % path)

    def hook_adjust_checkpoint(self, checkpoint):
        return checkpoint

    def save_checkpoint(self, epoch, iteration):
        raise NotImplementedError

    @property
    def divisor(self):
        return self.net.divisor

    @property
    def pad_input(self):
        return self._pad_input

class FeatureNet(SubNetwork):
    def __init__(self, device, args):
        super(FeatureNet, self).__init__(args, device)

        self.net = StereoUnaryUnetDyn(args.multi_level_output,
                                     args.activation, args.with_bn, args.with_upconv, args.with_output_bn)

        # provided pad-unpad
        self.pad_unpad = PadUnpad(self.net, self.net.divisor, tuple(args.pad))

        # load params
        self.load_parameters(args.checkpoint_unary, device)


class AffinityNet(SubNetwork):
    def __init__(self, device, args):
        super(AffinityNet, self).__init__(args, device)

        self.net = StereoUnaryUnetDyn(args.multi_level_output,
                                     args.activation, args.with_bn, args.with_upconv, args.with_output_bn)

        
        out_channel_factor = args.num_bp_layers
        out_channels = out_channel_factor * 2 * 5 # 2 directions, 5 values # L2-, L2+, L1-, L1+, L3 
        self.conv_out = [PaddedConv2d(ic, out_channels, 3, bias=True).to(device) for ic in \
                         self.net.out_channels]
        for lvl, conv in enumerate(self.conv_out):
            self.add_module('conv_out_lvl' + str(lvl), conv)

        # provide pad-unpad
        self.pad_unpad = PadUnpad(self.net, self.net.divisor, tuple(args.pad))
        self.to(device)

        # load params
        self.load_parameters(args.checkpoint_affinity, device)

    def forward(self, ipt):
        features = super(AffinityNet, self).forward(ipt)
        affinities = [torch.abs(conv(fi)) for conv, fi in zip(self.conv_out, features)] # outshape = N x 2 * 5 x H x W

        return affinities

class RefinementNet(SubNetwork):
    def __init__(self, device, args, in_channels=5, out_channels=1, with_output_relu=True):
        super(RefinementNet, self).__init__(args, device)

        self.net = []
        self.with_output_relu = with_output_relu

        for lvl in range(self.args.input_level_offset, self.args.output_level_offset - 1, -1):
            net = nn.Sequential(
                PaddedConv2d(in_channels, 32, 3, bias=True),
                ResidualBlock(32, 32, 3, dilation=1), 
                ResidualBlock(32, 32, 3, dilation=2), 
                ResidualBlock(32, 32, 3, dilation=4), 
                ResidualBlock(32, 32, 3, dilation=8), 
                ResidualBlock(32, 32, 3, dilation=1), 
                ResidualBlock(32, 32, 3, dilation=1), 
                PaddedConv2d(32, out_channels, 3, bias=True),
            )
            self.net.append(net)
            self.add_module('sn_' + str(lvl), net)

        self.to(device)
        self.relu = nn.ReLU(inplace=True)

        # load params
        self.load_parameters(args.checkpoint_refinement, device)


    def forward(self, I0_pyramid, d0, confidence, I1_pyramid):
        d0_lvl = d0.clone()
        refined_pyramid = []
        residuum_pyramid = []
        for ref_lvl in range(self.args.input_level_offset, self.args.output_level_offset -1, -1):
            I0_lvl = I0_pyramid[ref_lvl].to(self.device)
            I0_lvl = I0_lvl / I0_lvl.var(dim=(2,3), keepdim=True)

            # adapt input size
            scale_factor = I0_lvl.shape[2] / d0_lvl.shape[2]
            if abs(scale_factor - round(scale_factor)) > 0:
                print('WARNING: something weird is going on, got a fractional scale-factor in ref', scale_factor)
            d0_up = F.interpolate(d0_lvl.float(), size=I0_lvl.shape[2:], mode='nearest') * scale_factor
            conf_up = F.interpolate(confidence.float(), size=I0_lvl.shape[2:], mode='nearest')

            # compute input tensor
            ipt = torch.cat((I0_lvl, d0_up, conf_up), dim=1)
            residuum = self.net[ref_lvl - self.args.output_level_offset].forward(ipt)
            
            d0_lvl = d0_up.float() + residuum # flow
            if self.with_output_relu:
                d0_lvl = self.relu(d0_lvl) # stereo
            refined_pyramid.append(d0_lvl)
            residuum_pyramid.append(residuum)

        return refined_pyramid, [residuum_pyramid]
        

class EdgeNet(SubNetwork):
    def __init__(self, device, args):
        super(EdgeNet, self).__init__(args, device)

        self.net = StereoUnaryUnetDyn(args.multi_level_output, args.activation, args.with_bn, args.with_upconv, args.with_output_bn)

        out_channels = args.num_bp_layers * 2
        self.conv_out = [PaddedConv2d(ic, out_channels, 3, bias=True).to(device) for ic in \
                         self.net.out_channels]
        for lvl, conv in enumerate(self.conv_out):
            self.add_module('conv_out_lvl' + str(lvl), conv)

        # provide pad-unpad
        self.pad_unpad = PadUnpad(self.net, self.net.divisor, tuple(args.pad))
        self.to(device)

    def forward(self, ipt):
        features = super(EdgeNet, self).forward(ipt)
        edge_weights = [torch.abs(conv(fi)) for conv, fi in zip(self.conv_out, features)]
        return edge_weights


class PWNet(SubNetwork):
    def __init__(self, device, args):
        super(PWNet, self).__init__(args, device)

        self.net = StereoUnaryUnetDyn(args.multi_level_output,
                                     args.activation, args.with_bn, args.with_upconv, args.with_output_bn)

        out_channels = args.num_bp_layers * 2 * args.num_labels * args.num_labels # 2 directions, 5 values # L2-, L2+, L1-, L1+, L3 
      
        self.conv_out = [PaddedConv2d(ic, out_channels, 3, bias=True).to(device) for ic in \
                         self.net.out_channels]
        for lvl, conv in enumerate(self.conv_out):
            self.add_module('conv_out_lvl' + str(lvl), conv)

        # provide pad-unpad
        self.pad_unpad = PadUnpad(self.net, self.net.divisor, tuple(args.pad))
        self.to(device)

    def forward(self, ipt):
        features = super(PWNet, self).forward(ipt)

        pairwise_costs = [torch.abs(conv(fi)) for conv, fi in zip(self.conv_out, features)] # outshape = N x 2 * 5 x H x W

        return pairwise_costs