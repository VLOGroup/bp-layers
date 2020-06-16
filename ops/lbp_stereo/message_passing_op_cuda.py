import torch
import torch.nn as nn
import torch.nn.functional as F

from corenet import TemperatureSoftmin

import pytorch_cuda_lbp_op as lbp

def construct_pw_tensor(L1, L2, K):
    pw_tensor = torch.ones((K, K), dtype=torch.float) * L2

    for i in range(-1, 2):
        if i == 0:
            pw_tensor -= torch.diag(L2 * torch.ones(K - torch.abs(torch.tensor(i))), i)
        else:
            pw_tensor -= torch.diag((L2 - L1) * torch.ones(K - torch.abs(torch.tensor(i))), i)

    pw_tensor = pw_tensor.unsqueeze(0).unsqueeze(0)

    return pw_tensor

class LBPMinSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost, edge, messages, delta, affinities, offset):

        #print("Min sum forward OP......")

        #jump = construct_pw_tensor(L1, L2, cost.shape[3]).cuda()

        # affinities shape = N x 2*2 x 5 x H x W dim1 = 4 => already shifted
        # affinties dim1 = LR, RL, UD, DU
        # affinities dim2 = # L2-, L2+, L1-, L1+, L3 
        # affiniteis_input shape = N x 2*2 x 5+3 x H x W
        # affinities_input dim2 = # offset, L3, L2-, L1-, L0, L1+, L2+, L3 
        affinities_input = torch.zeros((affinities.shape[0], affinities.shape[1], affinities.shape[2] + 3, affinities.shape[3], affinities.shape[4])).cuda()
        
        affinities_input[:, :, 0, :, :] = offset      

        # L3 is symmetric => assign all directions
        affinities_input[:, :, 1, :, :] = affinities[:, :, -1, :, :]      
        affinities_input[:, :, -1, :, :] = affinities[:, :, -1, :, :]      

        # L0
        affinities_input[:, :, int(affinities_input.shape[2] / 2), :, :] = 0.0

        # everything else
        in_count = 0 # affinities counter
        for i in range(2, int(affinities_input.shape[2] / 2)): # i = affinities_input counter
            # copy original ordering for LR and UD (first negative, then positive)
            affinities_input[:, 0::2, i, :, :] = affinities[:, 0::2, in_count, :, :]  
            affinities_input[:, 0::2, -i, :, :] = affinities[:, 0::2, in_count + 1, :, :]   
            
            # copy reverse ordering for RL and DU (first positive, then negative)
            affinities_input[:, 1::2, i, :, :] = affinities[:, 1::2, in_count + 1, :, :]  
            affinities_input[:, 1::2, -i, :, :] = affinities[:, 1::2, in_count, :, :]   
            in_count += 2 

        #print(affinities_input[0, 0, :, 0, 0])

        affinities_input = affinities_input.contiguous()
        torch.cuda.empty_cache()
        messages, messages_argmin, message_scale = lbp.forward_minsum(cost, affinities_input, edge, messages, delta)
        
        ctx.save_for_backward(cost, affinities_input, edge, messages, messages_argmin, message_scale)
        return messages

    @staticmethod
    # @profile
    def backward(ctx, in_grad):
        cost, affinities_input, edge, messages, messages_argmin, message_scale = ctx.saved_tensors
       
        grad_cost, grad_affinities_input, grad_edge, grad_message = lbp.backward_minsum(cost, affinities_input, edge, in_grad.contiguous(), messages, messages_argmin, message_scale)

        #re-compute affinities grad for all learned params
        grad_affinities_out = torch.zeros((affinities_input.shape[0], affinities_input.shape[1], affinities_input.shape[2] - 3, affinities_input.shape[3], affinities_input.shape[4])).cuda()
       
        # sum up grad L3
        grad_affinities_out[:, :, -1, :, :] += grad_affinities_input[:, :, 1, :, :]
        grad_affinities_out[:, :, -1, :, :] += grad_affinities_input[:, :, -1, :, :]

        in_count = 0
        for i in range(2, int(affinities_input.shape[2] / 2)):
            grad_affinities_out[:, 0::2, in_count, :, :] = grad_affinities_input[:, 0::2, i, :, :]
            grad_affinities_out[:, 0::2, in_count + 1, :, :] = grad_affinities_input[:, 0::2, -i, :, :]  

            grad_affinities_out[:, 1::2, in_count + 1, :, :] = grad_affinities_input[:, 1::2, i, :, :]
            grad_affinities_out[:, 1::2, in_count, :, :] = grad_affinities_input[:, 1::2, -i, :, :]  
            in_count += 2

        #offset grad
        grad_offset = grad_affinities_input[:, :, 0, :, :]

        return grad_cost, grad_edge, grad_message, None, grad_affinities_out, grad_offset

class MessagePassing(nn.Module):

    @property
    def L1(self):
        return self._L1

    @property
    def L2(self):
        return self._L2

    def setL1L2(self, value_L1, value_L2):

        if value_L1 > 0 and value_L2 > 0 and value_L1 <= value_L2:
            self._L1 = value_L1
            self._L2 = value_L2
        elif value_L1 < 0 or value_L2 < 0:
            raise ValueError("L1 or L2 is < 0!")
        elif value_L1 > value_L2:
            raise ValueError("L1 must be smaller than or equal L2!")

    def __init__(self, device, max_iter, num_labels, delta, mode='min-sum'):
        super(MessagePassing, self).__init__()
        self.device = device
        self.max_iter = max_iter

        if mode != 'min-sum':
            raise ValueError("Unknown message parsing mode " + mode)
        self.mode = mode

        L1 = torch.tensor(0.1, device=device)
        L2 = torch.tensor(2.5, device=device)
        self._L1 = nn.Parameter(L1, requires_grad=True) 
        self._L2 = nn.Parameter(L2, requires_grad=True)

        self.softmin = TemperatureSoftmin(dim=3, init_temp=1.0)

        self.delta = delta

        self.rescaleT = None

    def projectL1L2(self):
        self.L2.data = torch.max(self.L1.data, self.L2.data)

    def forward(self, prob_vol, edge_weights, affinities, offset, messages):
        N, H, W, C = prob_vol.shape

        NUM_DIR = 4

        if edge_weights is None:
            edge_weights = torch.ones((N, NUM_DIR, H, W)).cuda()

        if affinities is None:
            if (self._L1 is not None) and (self._L2 is not None):
                # parameters are expected as follows L1-left L1-right L2
                affinities = torch.zeros((N, NUM_DIR, 3, H, W), dtype=torch.float).cuda()
                affinities[:, :, :2, :, :] = self._L1
                affinities[:, :, 2, :, :] = self._L2

        else:
            if (self._L1 is not None and self._L2 is None) or (self._L1 is None and self._L2 is not None):
                raise ValueError("L1 or L2 is None and affinities are not set!")

        if offset is None:
            offset = torch.zeros((N, NUM_DIR, H, W))

        if self.mode == 'min-sum':
            # convert to cost-input
            cost = -prob_vol

            # perform message-passing iterations
            for it in range(self.max_iter):
                torch.cuda.empty_cache()
                messages = LBPMinSumFunction.apply(cost, edge_weights, messages, self.delta, affinities, offset)
            
            # compute beliefs
            beliefs = messages.sum(dim=1) + cost

            # normalize output
            beliefs = self.softmin.forward(beliefs)
       
        else:
            raise NotImplementedError("message parsing mode " + self.mode + " is currently not implemented!")

        return beliefs, messages