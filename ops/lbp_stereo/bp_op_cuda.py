import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from networks import SubNetwork

import pytorch_cuda_lbp_op as lbp
from ops.lbp_stereo.inference_op import Inference
from ops.lbp_stereo.message_passing_op_cuda import MessagePassing

import numba

class BP(SubNetwork):

    @staticmethod
    def get_linear_idx(pos, L_vec_size):
        vec_idx = L_vec_size // 2 + pos
        vec_idx = np.maximum(1, vec_idx)
        vec_idx = np.minimum(L_vec_size - 1, vec_idx)

        return vec_idx
  
    @staticmethod
    def f_func(t, s, L_vec):

        o = L_vec[0]

        input_pos = t - s + o
        
        lower_pos = int(np.floor(input_pos))
        upper_pos = int(np.ceil(input_pos))

        if lower_pos == upper_pos:
            vec_idx = BP.get_linear_idx(lower_pos, L_vec.shape[0])
            #print(L_vec[vec_idx])
            return L_vec[vec_idx]

        lower_vec_idx = BP.get_linear_idx(lower_pos, L_vec.shape[0])
        upper_vec_idx = BP.get_linear_idx(upper_pos, L_vec.shape[0])

        lower_val = L_vec[lower_vec_idx]
        upper_val = L_vec[upper_vec_idx]

        weight_upper = input_pos - lower_pos
        weight_lower = upper_pos - input_pos

        interp_val = weight_lower * lower_val + weight_upper * upper_val
        
        return interp_val

    @staticmethod
    def construct_pw_energy_weights(K, L1, L2):
        '''K = num labels'''
        fij = np.ones((K, K)) * L2
        for i in range(-1, 2):
            if i == 0:
                fij -= np.diag(L2 * np.ones(K - np.abs(i)), i)
            else:
                fij -= np.diag((L2 - L1) * np.ones(K - np.abs(i)), i)

        return fij

    @staticmethod
    def construct_pw_prob_weights(K, L1, L2):
        '''K = num labels'''
        fij = np.zeros((K, K))

        for i in range(-1, 2):
            if i == 0:
                fij += np.diag(L1 * np.ones(K - np.abs(i)), i)
            else:
                fij += np.diag(L2 * np.ones(K - np.abs(i)), i)

        fij[fij == 0] = 1.0 - L1 - L2   #(1.0 - L1 - 2 * L2) / (K - 3.0) #1.0 - L1 - L2  

        return fij    

    # modes = wta / expectation
    def __init__(self, device, args, max_iter, num_labels, delta, mode_inference='expectation', mode_message_passing='min-sum', layer_idx=0, level=0):
        super(BP, self).__init__(args, device)
        self.device = device
        self.max_iter = max_iter
        self.layer_idx = layer_idx
        self.level = level
        self.delta = delta
        
        if mode_inference != 'wta' and mode_inference != 'expectation' and mode_message_passing != 'min-sum' and mode_inference != 'norm' and mode_inference != 'raw':
            raise ValueError("Unknown inference/message passing mode " + mode_inference + " " + mode_message_passing)

        self.message_passing = MessagePassing(self.device, self.max_iter, num_labels, delta, mode_message_passing)
        self.inference = Inference(self.device, mode_inference, mode_passing=mode_message_passing)
        
        if args.checkpoint_crf and args and 'checkpoint_crf' in args.__dict__.keys() and args.checkpoint_crf[0] is not None:
            self.load_parameters(args.checkpoint_crf[self.level][self.layer_idx], device)

    def forward(self, prob_vol, edge_weights, affinities = None, offsets = None):

        if len(prob_vol.shape) == 5:
            N, _, H, W, K = prob_vol.shape

            # u-flow        
            messages_u = torch.zeros((N, 4, H, W, K), requires_grad=True, device=self.device, dtype=torch.float)
            beliefs_u, messages_u = self.message_passing.forward(prob_vol[:,0].contiguous(), edge_weights, affinities, offsets, messages_u)
            result_u = self.inference.forward(beliefs_u)

            # v-flow        
            messages_v = torch.zeros((N, 4, H, W, K), requires_grad=True, device=self.device, dtype=torch.float)
            beliefs_v, messages_v = self.message_passing.forward(prob_vol[:,1].contiguous(), edge_weights, affinities, offsets, messages_u)
            result_v = self.inference.forward(beliefs_v)

            flow = torch.cat((result_u, result_v), dim=-1).permute(0,3,1,2)
            beliefs = torch.cat((beliefs_u.unsqueeze(1), beliefs_v.unsqueeze(1)), dim=1)
            messages = torch.cat((messages_u.unsqueeze(1), messages_v.unsqueeze(1)), dim=1)

            return flow, beliefs, messages
        else: # 4
            N, H, W, K = prob_vol.shape

            # disps        
            messages = torch.zeros((N, 4, H, W, K), requires_grad=True, device=self.device, dtype=torch.float)
            beliefs, messages = self.message_passing.forward(prob_vol.contiguous(), edge_weights, affinities, offsets, messages)
            result = self.inference.forward(beliefs)

            disps = result.permute(0,3,1,2)
            return disps, beliefs, messages


    def adjust_input_weights(self, weights, idx):
        if weights is not None:
            weights_idx = weights[:, idx * 2 : (idx + 1) * 2, :, :]

            # wx_L = np.zeros_like(wx)
            # wy_D = np.zeros_like(wy)
            # wx_L[:, 1:] = wx[:, :-1]
            # wy_D[1:, :] = wy[:-1, :]

            weights_input = torch.zeros((weights_idx.shape[0], 4, weights_idx.shape[2], weights_idx.shape[3])).cuda()

            weights_input[:, 0] = weights[:, 0]
            # wx RL
            weights_input[:, 1, :, 1:] = weights[:, 0, :, :-1]
            # wy UD
            weights_input[:, 2] = weights[:, 1]
            # wy DU
            weights_input[:, 3, 1:, :] = weights[:, 1, :-1, :]

            weights_input = weights_input.contiguous()

        else:
            weights_input = None

        return weights_input

    def adjust_input_affinities(self, affinities):
        # create affinities for 4 directions
        if affinities is not None:
            # outshape = N x 2, 5 x H x W
            # ensure ordering constraint # L2-, L2+, L1-, L1+, L3 
            # L3 >= L2 <= L1
            affinities_new = affinities.clone()
            affinities_new[:, :, 0] = torch.max(affinities[:, :, 0], affinities[:, :, 2])
            affinities_new[:, :, 1] = torch.max(affinities[:, :, 1], affinities[:, :, 3])
            affinities_new[:, :, 4] = torch.max(affinities[:, :, 4],
                                                torch.max(affinities_new[:, :, 0].clone(),
                                                        affinities_new[:, :, 1].clone()))

            affinities = affinities_new


            # shifted affinities
            affinities_shift = torch.zeros((affinities.shape[0], affinities.shape[1] + 2, affinities.shape[2], affinities.shape[3], affinities.shape[4])).cuda()
            
            # ax LR
            affinities_shift[:, 0] = affinities[:, 0]
            # ax RL 
            affinities_shift[:, 1, :, :, 1:] = affinities[:, 0, :, :, :-1]
            # ay UD
            affinities_shift[:, 2] = affinities[:, 1]
            # ay DU
            affinities_shift[:, 3, :, 1:, :] = affinities[:, 1, :, :-1, :]

            affinities_shift = affinities_shift.contiguous()   
        else:
            affinities_shift = None

        return affinities_shift

    def adjust_input_offsets(self, offsets):
        # create offsets for 4 directions
        if offsets is not None:
            # shifted offsets
            offsets_shift = torch.zeros((offsets.shape[0], offsets.shape[1] + 2, offsets.shape[2], offsets.shape[3])).cuda()
            
            # ox LR
            offsets_shift[:, 0] = offsets[:, 0]
            # ox RL
            offsets_shift[:, 1, :, 1:] = -offsets[:, 0, :, :-1]
            # oy UD
            offsets_shift[:, 2] = offsets[:, 1]
            # oy DU
            offsets_shift[:, 3, 1:, :] = -offsets[:, 1, :-1, :]

            offsets_shift = offsets_shift.contiguous()
        else:
            offsets_shift = None

        return offsets_shift


    def project_jumpcosts(self):
        self.message_passing.projectL1L2()

    def save_checkpoint(self, epoch, iteration):
        if 'c' in self.args.train_params:
            torch.save(self.state_dict(),
                       osp.join(self.args.train_dir, 'crf' + str(self.layer_idx) + '_lvl' + str(self.level) + '_checkpoint_' +
                                str(epoch) + '_' + str(iteration).zfill(6) + '.cpt'))

    def hook_adjust_checkpoint(self, checkpoint):
        # allow to continue a training where the temperature parameter after the BP did not exist
        if 'message_passing.softmin.T' not in checkpoint.keys():
            print('Info: Adjust loaded BP-checkpoint -> Use temperature T=1 (=fully backward compatible)')
            checkpoint['message_passing.softmin.T'] = self.message_passing.softmin.T

        if 'message_passing.L1' in checkpoint.keys() and 'message_passing.L2' in checkpoint.keys():
            print('Info: Adjust loaded BP-checkpoint -> Use setter for L1 L2')
            #L1 = nn.Parameter(checkpoint['message_passing.L1'])
            #L2 = nn.Parameter(checkpoint['message_passing.L2'])
            #self.setL1L2(L1, L2)
            checkpoint['message_passing._L1'] = checkpoint['message_passing.L1']
            checkpoint['message_passing._L2'] = checkpoint['message_passing.L2']
            checkpoint.pop('message_passing.L1')
            checkpoint.pop('message_passing.L2')

        return checkpoint

    @property
    def L1(self):
        return self.message_passing.L1

    @property
    def L2(self):
        return self.message_passing.L2

    def setL1L2(self, value_L1, value_L2):
        self.message_passing.setL1L2(value_L1, value_L2)

    @property
    def rescaleT(self):
        return self.message_passing.rescaleT

    @rescaleT.setter
    def rescaleT(self, value):
        self.message_passing.rescaleT = value



