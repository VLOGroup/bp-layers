import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FeatureNet, AffinityNet, RefinementNet
from matching import StereoMatchingSad
from ops.lbp_stereo.bp_op_cuda import BP
from ops.lbp_stereo.inference_op import Inference

from corenet import LrCheck, LrDistance, CvConfidence
from corenet import PadUnpad, Pad, Unpad

import numpy as np

class StereoMethod(nn.Module):
    def __init__(self, device, args):
        nn.Module.__init__(self)
        self.args = args

        self._feature_net = None
        self._matching = []
        self._affinity_net = None
        self._refinement_net = None
        self._crf = [] # list of bp layers

        max_dist = 3.0
        self.lr_check = LrCheck(device, max_dist).to(device)
        self.cv_conf = CvConfidence(device).to(device)
        self.lr_dist = LrDistance(device).to(device)

        self._min_disp = args.min_disp
        self._max_disp = args.max_disp

        self.pad = None
        self.unpad = None

        self.device = device

        self.logger = logging.getLogger("StereoMethod")


    def forward(self, I0_pyramid, I1_pyramid, offsets_orig=None, edges_orig=None, beliefs_in=None,
                min_disp=None, max_disp=None, step=None):
        
        # necessary for evaluation
        if self.pad is None:
            self.pad = Pad(self.feature_net.net.divisor, self.args.pad)
        if self.unpad is None:
            self.unpad = Unpad()

        res_dict = {'disps0': None}

        I0_in = I0_pyramid[self.args.input_level_offset].to(self.device)
        I1_in = I1_pyramid[self.args.input_level_offset].to(self.device)

        # pad input for multi-scale (for evaluation)
        I0_in = self.pad.forward(I0_in).cuda()
        I1_in = self.pad.forward(I1_in).cuda()

        f0_pyramid = self.extract_features(I0_in)
        f1_pyramid = self.extract_features(I1_in)

        if max_disp is not None:
            for matching_lvl, m in enumerate(self.matching):
                m.max_disp = ((max_disp + 1) // 2**matching_lvl) - 1
        if step is not None:
            for matching_lvl, m in enumerate(self.matching):
                m.step = step
        
        # multi-scale-matching
        prob_vol_pyramid = self.match(f0_pyramid, f1_pyramid)

        udisp0_pyramid = []
        for pv0 in prob_vol_pyramid:
            udisp0_pyramid.append(torch.argmax(pv0, dim=-1, keepdim=True).permute(0, 3, 1, 2))
        res_dict['disps0'] = udisp0_pyramid

        if self.args.model == 'wta':
            return res_dict

        affinity_pyramid = None
        if self.affinity_net:
            affinity_pyramid = self.extract_affinities(I0_in)
            for lvl in range(len(affinity_pyramid)):
                _, _, h, w = affinity_pyramid[lvl].shape
                affinity_pyramid[lvl] = affinity_pyramid[lvl].view((2, 5, h, w))
                affinity_pyramid[lvl] = affinity_pyramid[lvl].unsqueeze(0)
        
        output_disps_pyramid = []
        beliefs_pyramid = None
        crf_disps_pyramid = []
        beliefs_pyramid = []
        beliefs_in = None
        for lvl in reversed(range(len(prob_vol_pyramid))):
            pv_lvl = prob_vol_pyramid[lvl]
            m = self.matching[lvl]

            affinity = None
            if affinity_pyramid is not None:
                affinity = affinity_pyramid[lvl]
            crf = self.crf[lvl]

            # add probably an if condition whether do add multi-scale to crf
            if beliefs_in is not None:
                beliefs_in = F.interpolate(beliefs_in.unsqueeze(1), scale_factor=2.0, mode='trilinear')[:, 0]

                if beliefs_in.requires_grad:
                    # print('requires grad')
                    pv_lvl = pv_lvl / pv_lvl.sum(dim=-1, keepdim=True)
                else:
                    # print('no grad-> inplace')
                    pv_lvl += beliefs_in / 2.0 # in-place saves memory
                del beliefs_in

            torch.cuda.empty_cache()
            disps_lvl, beliefs_lvl, affinities_lvl, _ = self.optimize_crf(crf, pv_lvl, None, affinity)
            del affinities_lvl

            if lvl == 0:
                beliefs_lvl = self.unpad(beliefs_lvl, self.pad.l, self.pad.r, self.pad.t,
                                            self.pad.b, NCHW=False)
                disps_lvl = self.unpad(disps_lvl, self.pad.l, self.pad.r, self.pad.t,
                                        self.pad.b)

            beliefs_pyramid.append(beliefs_lvl)
            crf_disps_pyramid.append(disps_lvl + m.min_disp)
            beliefs_in = beliefs_pyramid[-1]

            # beliefs are from low res to high res
            beliefs_pyramid.reverse()
            crf_disps_pyramid.reverse()
            res_dict['disps0'] = crf_disps_pyramid

        if self.refinement_net:
            # crf
            cv_conf = self.cv_conf.forward(beliefs_pyramid[0].permute(0, 3, 1, 2), 
                                              crf_disps_pyramid[0])
            
            conf_all = cv_conf 
            refined_disps_pyramid, refinement_steps = self.refine_disps(I0_pyramid, 
                                                                        crf_disps_pyramid[0],
                                                                        confidence=conf_all,
                                                                        I1=I1_pyramid)
            if refinement_steps is not None:
                refinement_steps.reverse()
            refined_disps_pyramid.reverse()
            output_disps_pyramid = refined_disps_pyramid

            res_dict['disps0'] = output_disps_pyramid

        return res_dict

    def extract_features(self, ipt):
        if self.feature_net:
            return self.feature_net.forward(ipt)
        return None

    def extract_affinities(self, ipt):
        if self.affinity_net:
            return self.affinity_net.forward(ipt)
        return None

    def match(self, f0, f1, lr=False):
        prob_vols = []
        if self.matching:
            for matching, f0s, f1s in zip(self.matching, f0, f1):
                if lr:
                    f0s = torch.flip(f0s, dims=(3,)).contiguous()
                    f1s = torch.flip(f1s, dims=(3,)).contiguous()
                    prob_vol_s = matching.forward(f1s, f0s)
                    prob_vol_s = torch.flip(prob_vol_s, dims=(2,))
                else:
                    prob_vol_s = matching.forward(f0s, f1s)

                prob_vols.append(prob_vol_s)
            return prob_vols
        return None

    def optimize_crf(self, crf_layer, prob_vol, weights, affinities):
        if crf_layer:
            offsets = None
            # iterate over all bp "layers"
            for idx, crf in enumerate(crf_layer):
                prob_vol = prob_vol.contiguous()
                weights_input = crf.adjust_input_weights(weights, idx)
                affinities_shift = crf.adjust_input_affinities(affinities)
                offsets_shift = crf.adjust_input_offsets(offsets)

                if not prob_vol.requires_grad:
                    torch.cuda.empty_cache()

                disps, prob_vol, messages = crf.forward(prob_vol, weights_input, affinities_shift, offsets_shift)
                del messages # never used again

            return disps, prob_vol, affinities_shift, offsets_shift
        return None

    def refine_disps(self, I0, d0, confidence=None, I1=None):
        if self.refinement_net:
            refined, steps = self.refinement_net.forward(I0, d0, confidence, I1)
            return refined, steps
        return None

    def feature_net_params(self, requires_grad=None):
        if self.feature_net:
            return self.feature_net.parameter_list(requires_grad)
        return []

    def matching_params(self, requires_grad=None):
        params = []
        if self.matching:
            for m in self.matching:
                params += m.parameter_list(requires_grad)
        return params

    def affinity_net_params(self, requires_grad=None):
        if self.affinity_net:
            return self.affinity_net.parameter_list(requires_grad)
        return []

    def crf_params(self, requires_grad=None):
        crf_params = []
        if self.crf:
            for crf_layer in self.crf:
                for crf in crf_layer:
                    crf_params += crf.parameter_list(requires_grad)
        return crf_params

    def refinement_net_params(self, requires_grad=None):
        if self.refinement_net:
            return self.refinement_net.parameter_list(requires_grad)
        return []

    @property
    def feature_net(self):
        return self._feature_net

    @property
    def affinity_net(self):
        return self._affinity_net

    @property
    def crf(self):
        if self._crf == []:
            return None
        return self._crf

    @property
    def refinement_net(self):
        return self._refinement_net

    @property
    def matching(self):
        return self._matching

    @property
    def min_disp(self):
        return self._min_disp

    @property
    def max_disp(self):
        return self._max_disp

    @property
    def gc_net(self):
        return self._gc_net


####################################################################################################
# Block Match
####################################################################################################
class BlockMatchStereo(StereoMethod):
    def __init__(self, device, args):
        StereoMethod.__init__(self, device, args)
        self._feature_net = FeatureNet(device, args)

        self._matching = []
        for matching_lvl in range(self._feature_net.net.num_output_levels):
            if self.args.lbp_min_disp:
                min_disp = self.min_disp # original
            else:
                min_disp = self.min_disp // 2**matching_lvl
            max_disp = ((self.max_disp + 1) // 2**matching_lvl) - 1
            self.logger.info("Construct Matching Level %d with min-disp=%d and max-disp=%d" %(matching_lvl, min_disp, max_disp))

            self._matching.append(StereoMatchingSad(device, args, min_disp, max_disp, 
                                                    lvl=matching_lvl))


####################################################################################################
# Min-Sum LBP 
####################################################################################################
class MinSumStereo(BlockMatchStereo):
    def __init__(self, device, args):
        BlockMatchStereo.__init__(self, device, args)

        self.max_iter = args.max_iter
        num_labels = self.max_disp - self.min_disp + 1

        self._affinity_net = AffinityNet(device, args)

        for lvl in range(self._feature_net.net.num_output_levels):
            self._crf.append([BP(device, args, self.max_iter, num_labels, 3,
                        mode_inference = args.bp_inference,
                        mode_message_passing='min-sum', layer_idx=idx, level=lvl)
                    for idx in range(args.num_bp_layers)])


class RefinedMinSumStereo(MinSumStereo):
    def __init__(self, device, args):
        super(RefinedMinSumStereo, self).__init__(device, args)

        self._refinement_net = RefinementNet(device, args)