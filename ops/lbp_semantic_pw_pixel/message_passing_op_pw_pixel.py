
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_cuda_lbp_op as lbp

from corenet import TemperatureSoftmin

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
    def forward(ctx, cost, L1, L2, edge, messages, delta, jump):

        pw_net_jump_shift = torch.zeros((jump.shape[0] + 2, jump.shape[1], jump.shape[2], jump.shape[3], jump.shape[4])).cuda()

        # LR
        pw_net_jump_shift[0] = jump[0]
        # RL 
        pw_net_jump_shift[1, :, 1:] = jump[0, :, :-1].permute((0, 1, 3, 2))
        # UD
        pw_net_jump_shift[2] = jump[1]
        # DU
        pw_net_jump_shift[3, 1:] = jump[1, :-1].permute((0, 1, 3, 2))

        pw_net_jump_shift = pw_net_jump_shift.contiguous() 

        messages, messages_argmin, message_scale = lbp.forward_minsum(cost, pw_net_jump_shift, edge, messages, delta)
        
        ctx.save_for_backward(cost, edge, messages, messages_argmin, message_scale)
        return messages

    @staticmethod
    # @profile
    def backward(ctx, in_grad):
        cost, edge, messages, messages_argmin, message_scale = ctx.saved_tensors

        grad_cost, grad_jump, grad_edge, grad_message = lbp.backward_minsum(cost, edge, in_grad.contiguous(), messages, messages_argmin, message_scale)

        L1_grad = None
        L2_grad = None

        grad_jump[0, :, :-1] += grad_jump[1, :, 1:].permute((0, 1, 3, 2))
        grad_jump[1] = 0
        grad_jump[1] += grad_jump[2]
        grad_jump[1, :-1] += grad_jump[3, 1:].permute((0, 1, 3, 2))

        return grad_cost, L1_grad, L2_grad, grad_edge, grad_message, None, grad_jump[:2]

class MessagePassing(nn.Module):
    def __init__(self, device, max_iter, num_labels, delta, mode='min-sum'):
        super(MessagePassing, self).__init__()

        self.device = device
        self.max_iter = max_iter

        if mode != 'min-sum':
            raise ValueError("Unknown message parsing mode " + mode)
        self.mode = mode

        L1 = torch.tensor(0.1, device=device)
        L2 = torch.tensor(2.5, device=device)
        self.L1 = nn.Parameter(L1, requires_grad=True) 
        self.L2 = nn.Parameter(L2, requires_grad=True)

        self.softmin = TemperatureSoftmin(dim=3, init_temp=1.0)

        self.delta = delta
        self.rescaleT = None

    def projectL1L2(self):
        self.L2.data = torch.max(self.L1.data, self.L2.data)

    def forward(self, prob_vol, edge_weights, messages, jump):

        N, H, W, C = prob_vol.shape
        if edge_weights is None:
            edge_weights = torch.ones((N, 4, H, W))

        if self.mode == 'min-sum':
            # convert to cost-input
            cost = -prob_vol

            # perform message-passing iterations
            for it in range(self.max_iter):
                messages = LBPMinSumFunction.apply(cost, self.L1, self.L2, edge_weights, messages, self.delta, jump)
            
            # compute beliefs
            beliefs = messages.sum(dim=1) + cost

            # normalize output
            beliefs = self.softmin.forward(beliefs)

        else:
            raise NotImplementedError("message parsing mode " + self.mode + " is currently not implemented!")

        return beliefs