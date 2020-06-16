import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import FeatureNet, AffinityNet, RefinementNet
from flow_matching import FlowMatchingSad
from ops.lbp_stereo.bp_op_cuda import BP

from corenet import CvConfidence#, LrCheck, LrDistance,
from corenet import Pad, Unpad

class FlowMethod(nn.Module):
    def __init__(self, device, args):
        nn.Module.__init__(self)
        self.args = args

        self._feature_net = None
        self._matching = []
        self._affinity_net = None
        self._refinement_net = None
        self._crf = [] # list of bp layers

        self.cv_conf = CvConfidence(device).to(device)

        self._sws = args.sws

        self.pad = None
        self.unpad = None

        self.device = device

    def forward(self, I0_pyramid, I1_pyramid, beliefs_in=None, sws=None):
        
        # necessary for evaluation
        if self.pad is None:
            self.pad = Pad(self.feature_net.net.divisor, self.args.pad)
        if self.unpad is None:
            self.unpad = Unpad()

        res_dict = {'flow0': None}

        I0_in = I0_pyramid[self.args.input_level_offset].to(self.device)
        I1_in = I1_pyramid[self.args.input_level_offset].to(self.device)

        # pad input for multi-scale (for evaluation)
        I0_in = self.pad.forward(I0_in)
        I1_in = self.pad.forward(I1_in)

        f0_pyramid = self.extract_features(I0_in)
        f1_pyramid = self.extract_features(I1_in)

        # if sws is not None:
        #     self.matching.sws = sws

        # multi-scale-matching
        prob_vol_pyramid = self.match(f0_pyramid, f1_pyramid)

        uflow0_pyramid = []
        for pv0 in prob_vol_pyramid:
            uflow0_pyramid.append(torch.argmax(pv0, dim=-1))
        res_dict['flow0'] = uflow0_pyramid

        if not self._crf:
            return res_dict

        affinity_pyramid = None
        if self.affinity_net:
            affinity_pyramid = self.extract_affinities(I0_in)
            for lvl in range(len(affinity_pyramid)):
                _, _, h, w = affinity_pyramid[lvl].shape
                affinity_pyramid[lvl] = affinity_pyramid[lvl].view((-1, 2, 5, h, w))
                affinity_pyramid[lvl] = affinity_pyramid[lvl].unsqueeze(0)
        
        output_flow_pyramid = []
        beliefs_pyramid = None

        crf_flow_pyramid = []
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
                N,_,H,W,K = beliefs_in.shape
                size = (2*H, 2*W, 2*K-1)
                beliefs_in_u = F.interpolate(beliefs_in[:, 0].unsqueeze(1), size=size, mode='trilinear')[:, 0]
                beliefs_in_v = F.interpolate(beliefs_in[:, 1].unsqueeze(1), size=size, mode='trilinear')[:, 0]
                beliefs_in = torch.cat((beliefs_in_u.unsqueeze(1), beliefs_in_v.unsqueeze(1)), dim=1).contiguous()
                pv_lvl = pv_lvl + beliefs_in / 2.0

            flow_lvl, beliefs_lvl, affinities_lvl, offsets_lvl = self.optimize_crf(crf, pv_lvl, None, affinity, None)

            if lvl == 0: # TODO FOR EVAL!!!!
                beliefs_lvl = self.unpad(beliefs_lvl, self.pad.l, self.pad.r, self.pad.t,
                                            self.pad.b, NCHW=False)
                flow_lvl = self.unpad(flow_lvl, self.pad.l, self.pad.r, self.pad.t,
                                        self.pad.b)

            beliefs_pyramid.append(beliefs_lvl)
            crf_flow_pyramid.append(flow_lvl - m.sws // 2)

            beliefs_in = beliefs_pyramid[-1]

        # beliefs are from low res to high res
        beliefs_pyramid.reverse()
        crf_flow_pyramid.reverse()
        output_flow_pyramid = crf_flow_pyramid
        res_dict['flow0'] = crf_flow_pyramid
           
        if self.refinement_net:
            # crf
            cv_conf_u = self.cv_conf.forward(beliefs_pyramid[0][:,0].permute(0, 3, 1, 2), 
                                              crf_flow_pyramid[0][:,0:1] +  m.sws // 2)
            cv_conf_v = self.cv_conf.forward(beliefs_pyramid[0][:,1].permute(0, 3, 1, 2), 
                                              crf_flow_pyramid[0][:,1:2] +  m.sws // 2)
            
            conf_all =  torch.cat((cv_conf_u, cv_conf_v), dim=1)
            refined_flow_pyramid, _ = self.refine_disps(I0_pyramid, 
                                                                        crf_flow_pyramid[0],
                                                                        confidence=conf_all,
                                                                        I1=I1_pyramid)
            refined_flow_pyramid.reverse()
            output_flow_pyramid = refined_flow_pyramid

            res_dict['flow0'] = output_flow_pyramid

        return res_dict

    def extract_features(self, ipt):
        if self.feature_net:
            return self.feature_net.forward(ipt)
        return None

    def compute_guidance(self, ipt):
        if self.guidance_net:
            return self.guidance_net.forward(ipt)
        return None

    def extract_edges(self, ipt):
        if self.edge_net:
            return self.edge_net.forward(ipt)
        return None

    def extract_affinities(self, ipt):
        if self.affinity_net:
            return self.affinity_net.forward(ipt)
        return None

    def extract_offsets(self, ipt):
        if self.offset_net:
            return self.offset_net.forward(ipt)
        return None

    def match(self, f0, f1):
        prob_vols = []
        if self.matching:
            for matching, f0s, f1s in zip(self.matching, f0, f1):
                prob_vols.append(matching.forward(f0s, f1s))
            return prob_vols
        return None

    def optimize_crf(self, crf_layer, prob_vol, weights, affinities, offsets):
        if crf_layer:
            # iterate over all bp "layers"
            for idx, crf in enumerate(crf_layer):
                #TODO take care of BP layer idx in adjust functions
                prob_vol = prob_vol.contiguous()
                weights_input = crf.adjust_input_weights(weights, idx)
                affinities_shift = crf.adjust_input_affinities(affinities[:,idx])
                offsets_shift = crf.adjust_input_offsets(offsets)

                disps, prob_vol, messages = crf.forward(prob_vol, weights_input, affinities_shift, offsets_shift)

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
    def offset_net(self):
        return self._offset_net

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

   
####################################################################################################
# Block Match
####################################################################################################
class BlockMatchFlow(FlowMethod):
    def __init__(self, device, args):
        FlowMethod.__init__(self, device, args)
        self._feature_net = FeatureNet(device, args)

        self._matching = []
        for matching_lvl in range(self._feature_net.net.num_output_levels):
            sws = ((self._sws) // 2**matching_lvl)
            self._matching.append(FlowMatchingSad(device, args, sws, lvl=matching_lvl))
            if args.matching != 'sad':
                print('WARNING: Use SAD matching for flow, but', args.matching, 'was chosen.')


####################################################################################################
# Min-Sum LBP
####################################################################################################
class MinSumFlow(BlockMatchFlow):
    def __init__(self, device, args):
        BlockMatchFlow.__init__(self, device, args)

        self.max_iter = args.max_iter
        num_labels = self._sws + 1

        self._affinity_net = AffinityNet(device, args)

        for lvl in range(self._feature_net.net.num_output_levels):
            self._crf.append([BP(device, args, self.max_iter, num_labels, 3,
                        mode_inference = args.bp_inference,
                        mode_message_passing='min-sum', layer_idx=idx, level=lvl)
                    for idx in range(args.num_bp_layers)])

class RefinedMinSumFlow(MinSumFlow):
    def __init__(self, device, args):
        super(RefinedMinSumFlow, self).__init__(device, args)

        self._refinement_net = RefinementNet(device, args, in_channels=7, out_channels=2, with_output_relu=False)
