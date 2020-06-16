import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from networks import SubNetwork
from networks import PWNet, EdgeNet

from ops.lbp_semantic_pw.bp_op_cuda import BP as BP_PW
from ops.lbp_semantic_pw_pixel.bp_op_cuda import BP as BP_PW_PIXEL

from dependencies.ESPNet.test.Model import ESPNet

import os

from corenet import TemperatureSoftmax

from time import time

class SemanticNet(SubNetwork):
    def __init__(self, device, args):
        super(SemanticNet, self).__init__(args, device)

        self.num_labels = args.num_labels

        self.forward_timings = []

        self._esp_net = ESPNet(self.num_labels, 2, 8, None)

        self._softmax_input = TemperatureSoftmax(dim=3, init_temp=1.0)

        if (self.args.pairwise_type == "global") and args.with_edges:
            self._edge_net = EdgeNet(device, args)

        if self.args.checkpoint_esp_net:
            esp_weight_file = os.path.join(self.args.checkpoint_esp_net, "decoder/espnet_p_" + str(2) + "_q_" + str(8) + ".pth")
            self._esp_net.load_state_dict(torch.load(esp_weight_file))

        max_iter = 1
      
        if args.pairwise_type == "global":
            self._crf = [BP_PW(device, args, max_iter, self.num_labels, self.num_labels,
                    mode_inference = 'wta',
                    mode_message_passing='min-sum', layer_idx=0)]

            self.pw_mat_init = torch.zeros((1, 2, self.num_labels, self.num_labels), dtype=torch.float).cuda()
            self._pw_mat = nn.Parameter(self.pw_mat_init, requires_grad=True) 
            self._pw_net = None

        elif args.pairwise_type == "pixel":
            self._crf = [BP_PW_PIXEL(device, args, max_iter, self.num_labels, self.num_labels,
                    mode_inference = 'wta',
                    mode_message_passing='min-sum', layer_idx=0)]

            self._pw_net = PWNet(device, args)
            self._pw_mat = None

        if args.checkpoint_semantic is not None:
            print("Loading semantic checkpoint!")
            self.load_state_dict(torch.load(args.checkpoint_semantic))   

        if not self.args.with_esp:
            self._esp_net.eval()

        self._esp_net.to(device)

    def forward(self, ipt):

        t0_fwd = time()

        res_esp = self._esp_net.forward(ipt)

        res_esp = res_esp.permute((0, 2, 3, 1))
        res_esp = res_esp.contiguous()

        res_esp = self._softmax_input(res_esp)

        esp_only_res = res_esp[0].max(2)[1].unsqueeze(0).unsqueeze(0)

        N, H, W, C = res_esp.shape
        
        if (self.args.pairwise_type == "global") and self.args.with_edges:
            weights = self.extract_edges(ipt)[0]
        else:
            weights = torch.ones((N, 2, H, W), dtype=torch.float).cuda()
       
        res, beliefs = self.optimize_crf(ipt, res_esp, weights, None, None)
        
        torch.cuda.synchronize()
        self.forward_timings.append(time() - t0_fwd)  

        torch.cuda.empty_cache()

        return res, beliefs, esp_only_res

    def esp_net(self):
        return self._esp_net
    
    def esp_net_params(self, requires_grad=None):
        if self._esp_net:
            return self._esp_net.parameter_list(requires_grad)
        return []   

    def sem_params(self, requires_grad=None):
        return self.parameter_list(requires_grad)
       
    def pw_mat(self):
        return self._pw_mat

    def edge_net(self):
        return self._edge_net
    
    def edge_net_params(self, requires_grad=None):
        if self._edge_net:
            return self._edge_net.parameter_list(requires_grad)
        return []

    def extract_edges(self, ipt):
        if self.edge_net:
            return self._edge_net.forward(ipt)
        return None        

    def optimize_crf(self, ipt, prob_vol, weights, affinities, offsets):
        # iterate over all bp "layers"
        for idx, crf in enumerate(self._crf):

            prob_vol = prob_vol.contiguous()

            weights_input = crf.adjust_input_weights(weights, idx)

            if self._pw_net is not None:

                pw_net_jump = self._pw_net.forward(ipt)[0]

                _, _, H, W = pw_net_jump.shape

                pw_net_jump = pw_net_jump.view((2, self.num_labels, self.num_labels, H, W))
                pw_net_jump = pw_net_jump.permute((0, 3, 4, 1, 2)).contiguous() 

                disps, prob_vol = crf.forward(prob_vol, weights_input, pw_net_jump)
            elif self._pw_mat is not None:
                disps, prob_vol = crf.forward(prob_vol, weights_input, self._pw_mat)

        return disps, prob_vol
