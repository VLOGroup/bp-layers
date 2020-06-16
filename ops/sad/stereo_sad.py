import torch

import pytorch_cuda_stereo_sad_op

class StereoMatchingSadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f1, min_disp, max_disp, step=1.0):
        ctx.save_for_backward(f0, f1, torch.tensor(min_disp), torch.tensor(max_disp), torch.tensor(step))
        res = pytorch_cuda_stereo_sad_op.forward(f0, f1, min_disp, max_disp, step)
        return res

    @staticmethod
    def backward(ctx, in_grad):
        f0, f1, min_disp, max_disp, step = ctx.saved_tensors
        if step != 1.0:
            raise ValueError("Error: Backward for step != 1 is not implemented!")
        df0, df1 = pytorch_cuda_stereo_sad_op.backward(f0, f1, int(min_disp), int(max_disp), 
                                                       in_grad)
        return df0, df1, None, None
