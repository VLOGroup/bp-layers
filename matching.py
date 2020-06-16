import torch
from networks import SubNetwork
from corenet import TemperatureSoftmax

from ops.sad.stereo_sad import StereoMatchingSadFunction

class StereoMatching(SubNetwork):
    def __init__(self, device, args, min_disp, max_disp, in_channels=None, lvl=0, step=1.0):
        super(StereoMatching, self).__init__(args, device)
        self.device = device
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.step = step
        self.in_channels = in_channels
        self.level = lvl

        self._softmax = TemperatureSoftmax(dim=3, init_temp=1.0)

    def compute_score_volume(self, f0, f1):
        raise NotImplementedError

    def forward(self, f0, f1):
        score_vol = self.compute_score_volume(f0, f1)
        prob_vol = self._softmax.forward(score_vol)
        return prob_vol.contiguous()

    @staticmethod
    def argmin_to_disp(argmin, min_disp):
        res = argmin + min_disp
        return res

    def save_checkpoint(self, epoch, iteration):
        pass

class StereoMatchingSad(StereoMatching):
    def __init__(self, device, args, min_disp, max_disp, lvl=0, step=1.0):
        super(StereoMatchingSad, self).__init__(device, args, min_disp, max_disp, lvl=lvl, step=step)

        self.load_parameters(args.checkpoint_matching[self.level], device)
        self.to(device)

    def compute_score_volume(self, f0, f1):
        cost_vol = StereoMatchingSadFunction.apply(f0, f1, self.min_disp, self.max_disp, self.step)
        return -cost_vol