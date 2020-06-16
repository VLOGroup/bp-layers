import os.path as osp
import torch

from networks import SubNetwork

from corenet import TemperatureSoftmax
from ops.flow_mp_sad.flow_mp_sad import FlowMpSadFunction

class FlowMatching(SubNetwork):
    def __init__(self, device, args, sws, in_channels=None, lvl=0):
        super(FlowMatching, self).__init__(args, device)
        self.device = device
        self.sws = sws
        self.in_channels = in_channels
        self.level = lvl

        self._softmax = TemperatureSoftmax(dim=3, init_temp=1.0)

    def compute_score_volume(self, f0, f1):
        raise NotImplementedError

    def forward(self, f0, f1):
        score_vol_u, score_vol_v = self.compute_score_volume(f0, f1)
        prob_vol_u = self._softmax.forward(score_vol_u)
        prob_vol_v = self._softmax.forward(score_vol_v)
        prob_vol_uv = torch.cat((prob_vol_u.unsqueeze(1), prob_vol_v.unsqueeze(1)), dim=1)
        return prob_vol_uv.contiguous()

    @staticmethod
    def argmin_to_disp(argmin, min_disp):
        res = argmin + min_disp
        return res

    def save_checkpoint(self, epoch, iteration):
        if 'u' in self.args.train_params:
            torch.save(self.state_dict(),
                       osp.join(self.args.train_dir, 'matching_lvl' + str(self.level) +
                       '_checkpoint_' +  str(epoch) + '_' + str(iteration).zfill(6) + '.cpt'))


class FlowMatchingSad(FlowMatching):
    def __init__(self, device, args, sws, lvl=0):
        super(FlowMatchingSad, self).__init__(device, args, sws, lvl=lvl)

        if args.checkpoint_matching: # not-empty check
            lvl = min(self.level, len(args.checkpoint_matching) - 1)
            self.load_parameters(args.checkpoint_matching[lvl], device)
        self.to(device)

    def compute_score_volume(self, f0, f1):
        cv_u, cv_v, amin_u, amin_v = FlowMpSadFunction.apply(f0, f1, self.sws)
        return -cv_u, -cv_v
