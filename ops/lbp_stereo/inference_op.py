
import torch
import torch.nn as nn
import numpy as np

import pytorch_cuda_lbp_op as lbp

class Inference(nn.Module):
    # modes = wta / expectation
    def __init__(self, device, mode='wta', mode_passing='min-sum'):
        super(Inference, self).__init__()
        self.device = device
        if      mode != 'wta' \
            and mode != 'expectation' \
            and mode != 'norm' \
            and mode != 'raw' \
            and mode != 'sub-exp':
            raise ValueError("Unknown inference mode " + mode)
        self.mode = mode
        self.mode_passing = mode_passing

    def forward(self, beliefs):
        if self.mode == "wta" and self.mode_passing == "min-sum":
            res = torch.argmax(beliefs, dim=3, keepdim=True).float()

        if self.mode == "wta":
            res = torch.argmax(beliefs, dim=3, keepdim=True)    
        elif self.mode == "expectation":
            beliefs_normal = beliefs / beliefs.sum(dim=3, keepdim=True)

            if torch.isnan(beliefs_normal).sum() > 0:
                print("Beliefs normalized contains " + str(torch.isnan(beliefs_normal).sum()) + \
                      " NaNs ;(")

            labels = np.arange(beliefs.shape[3])[np.newaxis, np.newaxis, np.newaxis, :]
            labels_tensor = torch.tensor(labels.astype('float32'), device=self.device)
            res = (beliefs_normal * labels_tensor).sum(dim=3, keepdim=True)

        elif self.mode == "norm":
            beliefs_normal = beliefs / beliefs.sum(dim=3, keepdim=True)
            res = beliefs_normal

        elif self.mode == "raw":
            #print("using raw inference...")
            res = beliefs

        elif self.mode == 'sub-exp':
            res = self.compute_sub_expectation(beliefs)

        return res

    @staticmethod
    def compute_sub_expectation(beliefs, support=3):
        N, H, W, K = beliefs.shape
        device = beliefs.device

        disp = beliefs.argmax(dim=-1).unsqueeze(-1)

        # generate coordinates
        n_coords = torch.arange(N, device=device, dtype=torch.long)
        n_coords = n_coords.view(-1, 1, 1, 1)

        x_coords = torch.arange(W, device=device, dtype=torch.long).view(1, 1, -1, 1)
        y_coords = torch.arange(H, device=device, dtype=torch.long).view(1, -1, 1, 1)

        # nl = n_coords.expand((N, H, W, K)).long()
        # xl = x_coords.expand((N, H, W, K)).long()
        # yl = y_coords.expand((N, H, W, K)).long()

        #disp_multiple_hot = torch.zeros((N, H, W, K), device=device, dtype=torch.float)
        #torch.cuda.empty_cache()
        #for offset in range(-support, support + 1):
        #    disp_offs = torch.min(torch.max(disp + offset, torch.tensor(0, device=device)), 
        #                          torch.tensor(K - 1, device=device))
        #    print(offset)
        #    disp_multiple_hot[nl, yl, xl, disp_offs] = 1
        #    torch.cuda.empty_cache()

        # disps_range = torch.arange(K, device=device, dtype=torch.float).view(1, 1, 1, -1)

        # beliefs_max = beliefs * disp_multiple_hot
        # beliefs_max_normalized = beliefs_max / beliefs_max.sum(dim=-1, keepdim=True) 
        # disp_subpix = torch.sum(beliefs_max_normalized * disps_range, dim=-1, keepdim=True)

        # reduces GPU memory requirement significantly
        ws = 2 * support + 1
        nl = n_coords.expand((N, H, W, ws)).long()
        xl = x_coords.expand((N, H, W, ws)).long()
        yl = y_coords.expand((N, H, W, ws)).long()

        disp_windows = torch.arange(-support, support + 1).cuda() + disp
        disp_windows = torch.min(torch.max(disp_windows, torch.tensor(0, device=device)), torch.tensor(K - 1, device=device))
        beliefs_windows = beliefs[nl, yl, xl, disp_windows]
        beliefs_windows_normalized = beliefs_windows / beliefs_windows.sum(dim=-1, keepdim=True)
        disp_subpix = torch.sum(beliefs_windows_normalized * disp_windows, dim=-1, keepdim=True)

        return disp_subpix
