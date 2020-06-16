import argparse
import torch

from flow import MinSumFlow, BlockMatchFlow, RefinedMinSumFlow
import data
from flow_tools import computeFlowImg

import imageio 
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--im0', action='store', default="data/sf_0006_left.png", required=False, type=str)
parser.add_argument('--im1', action='store', default="data/sf_0006_right.png", required=False, type=str)
parser.add_argument('--sws', action='store', default=108, type=int)
parser.add_argument('--multi-level-output', action='store_true', default=False)
parser.add_argument('--activation', action='store', choices=['relu', 'leakyrelu', 'elu'], default='leakyrelu')
parser.add_argument('--with-bn', action='store_true', default=False)
parser.add_argument('--with-upconv', action='store_true', default=False)
parser.add_argument('--with-output-bn', action='store_true', default=False)
parser.add_argument('--pad', action='store', default=(0, 0), nargs=2, type=int,
                        help='extra padding of in height and in width on every side')

parser.add_argument('--model', action='store', default='bp+ms+ref+h', 
                    choices=['wta', 'bp+ms', 'bp+ms+h', 'bp+ms+ref+h'])
parser.add_argument('--checkpoint-unary', action='store', default=None, type=str)
parser.add_argument('--checkpoint-matching', action='store', default=[], nargs='+', type=str)
parser.add_argument('--checkpoint-affinity', action='store', default=None, type=str)
parser.add_argument('--checkpoint-crf', action='append', default=[], type=str, nargs='+')
parser.add_argument('--checkpoint-refinement', action='store', default=None, type=str)

parser.add_argument('--lbp-min-disp', action='store_true', default=False)
parser.add_argument('--max-iter', action='store', default=1, type=int)
parser.add_argument('--num-bp-layers', action='store', default=1, type=int)
parser.add_argument('--bp-inference', action='store', default='sub-exp',
                        choices=['wta', 'expectation', 'sub-exp'], type=str)

parser.add_argument('--matching', action='store', choices=['corr', 'sad', 'conv3d'],
                        default='sad', type=str)

parser.add_argument('--input-level-offset', action='store', default=1, type=int,
                    help='1 means that level 1 is the input resolution')
parser.add_argument('--output-level-offset', action='store', default=1, type=int,
                    help="0 means that level 0 (=full res) is the output resolution")                        
args = parser.parse_args()

I0_pyramid, I1_pyramid = data.load_sample(args.im0, args.im1)

args.multi_level_output = True

device = 'cuda:0'
with torch.no_grad():
    if args.model == 'wta':
        model = BlockMatchFlow(device, args)
    elif args.model == 'bp+ms':
        model = MinSumFlow(device, args)
    elif args.model == 'bp+ms+h':
        model = MinSumFlow(device, args)
    elif args.model == 'bp+ms+ref+h':
        model = RefinedMinSumFlow(device, args)

    res_dict = model.to(device).forward(I0_pyramid, I1_pyramid, sws=args.sws)


flow = res_dict['flow0'][0].squeeze().float().detach().cpu().numpy()

gt_flow = data.readFlow("data/frame_0019.flo") # only used for normalizing the flow output!
flow_img = computeFlowImg(flow[0], flow[1], gt_flow[0], gt_flow[1])

imageio.imwrite("data/output/flow/" + args.model + ".png", flow_img)