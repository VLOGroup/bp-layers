import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from networks import SubNetwork

from ops.lbp_semantic_pw_pixel.inference_op import Inference
from ops.lbp_semantic_pw_pixel.message_passing_op_pw_pixel import MessagePassing

class BP(SubNetwork):
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
    def __init__(self, device, args, max_iter, num_labels, delta, mode_inference='expectation', mode_message_passing='min-sum', layer_idx=0):
        super(BP, self).__init__(args, device)
        self.device = device
        self.max_iter = max_iter
        self.layer_idx = layer_idx
        self.delta = delta

        print("init pixel wise bp...")
        
        if mode_inference != 'wta' and mode_inference != 'expectation' and mode_message_passing != 'min-sum'  and mode_inference != 'norm' and mode_inference != 'raw':
            raise ValueError("Unknown inference/message passing mode " + mode_inference + " " + mode_message_passing)

        self.message_passing = MessagePassing(self.device, self.max_iter, num_labels, self.delta, mode_message_passing)
        self.inference = Inference(self.device, mode_inference, mode_passing=mode_message_passing)
        

    def forward(self, prob_vol, edge_weights, jump):

        N, H, W, K = prob_vol.shape
        messages = torch.zeros((N, 4, H, W, K), requires_grad=False, device=self.device,
                               dtype=torch.float)

        # compute messages
        beliefs = self.message_passing.forward(prob_vol, edge_weights, messages, jump)

        #  + wta/expectation
        result = self.inference.forward(beliefs)

        return result.permute(0,3,1,2), beliefs

    def project_jumpcosts(self):
        self.message_passing.projectL1L2()

    def save_checkpoint(self, epoch, iteration):
        if 'c' in self.args.train_params:
            torch.save(self.state_dict(),
                       osp.join(self.args.train_dir, 'crf' + str(self.layer_idx) + '_checkpoint_' +
                                str(epoch) + '_' + str(iteration).zfill(6) + '.cpt'))

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

    @property
    def L1(self):
        return self.message_passing.L1

    @L1.setter
    def L1(self, value):
        self.message_passing.L1.data = torch.tensor(value, device=self.device, dtype=torch.float)

    @property
    def L2(self):
        return self.message_passing.L2

    @L2.setter
    def L2(self, value):
        self.message_passing.L2.data = torch.tensor(value, device=self.device, dtype=torch.float)

    @property
    def rescaleT(self):
        return self.message_passing.rescaleT

    @rescaleT.setter
    def rescaleT(self, value):
        self.message_passing.rescaleT = value



