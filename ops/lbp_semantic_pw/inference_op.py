
import torch
import torch.nn as nn
import numpy as np

import pytorch_cuda_lbp_op as lbp

class Inference(nn.Module):
    # modes = wta / expectation
    def __init__(self, device, mode='wta', mode_passing='min-sum'):
        super(Inference, self).__init__()
        self.device = device
        if mode != 'wta' and mode != 'expectation' and mode != 'norm' and mode != 'raw':
            raise ValueError("Unknown inference mode " + mode)
        self.mode = mode
        self.mode_passing = mode_passing

    def forward(self, beliefs):
        if self.mode == "wta" and self.mode_passing == "min-sum":
            res = torch.argmax(beliefs, dim=3, keepdim=True)
        if self.mode == "wta":
            res = torch.argmax(beliefs, dim=3, keepdim=True)    
        elif self.mode == "expectation":
            beliefs_normal = beliefs / beliefs.sum(dim=3, keepdim=True)

            if torch.isnan(beliefs_normal).sum() > 0:
                print("Beliefs normalized contains " + str(torch.isnan(beliefs_normal).sum()) + " NaNs ;(")

            labels = np.arange(beliefs.shape[3])[np.newaxis, np.newaxis, np.newaxis, :]
            labels_tensor = torch.tensor(labels.astype('float32'), device=self.device)
            res = (beliefs_normal * labels_tensor).sum(dim=3, keepdim=True)
        elif self.mode == "norm":
            beliefs_normal = beliefs / beliefs.sum(dim=3, keepdim=True)
            res = beliefs_normal
        elif self.mode == "raw":
            #print("using raw inference...")
            res = beliefs

        return res
