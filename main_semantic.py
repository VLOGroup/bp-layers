from semantic_segmentation import SemanticNet

import argparse
import os
import yaml
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from imageio import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--img', action='store', required=True, type=str)
parser.add_argument('--scale-imgs', action='store', default=0.5, type=float)
parser.add_argument('--checkpoint-semantic', action='store', default=None, type=str)
parser.add_argument('--checkpoint-esp-net', action='store', default=None, type=str)
parser.add_argument('--num-labels', action='store', default=20, type=int)
parser.add_argument('--activation', action='store', choices=['relu', 'leakyrelu', 'elu'], default='leakyrelu')    
parser.add_argument('--num-bp-layers', action='store', default=1, type=int)      
parser.add_argument('--with-bn', action='store_true', default=True)          
parser.add_argument('--with-upconv', action='store_true', default=False)
parser.add_argument('--with-output-bn', action='store_true', default=False)
parser.add_argument('--pad', action='store', default=(0, 0), nargs=2, type=int, help='extra padding of in height and in width on every side')
parser.add_argument('--pairwise-type', action='store', choices=["global", "pixel"], default="global")
parser.add_argument('--multi-level-features', action='store_true', default=False)
parser.add_argument('--with-esp', action='store_true', default=False)
parser.add_argument('--with-edges', action='store_true', default=False)
parser.add_argument('--multi-level-output', action='store_true', default=False)


args = parser.parse_args()

cuda_device = 'cuda:0'

semantic_model = SemanticNet(cuda_device, args)

# read input image
test_img_path = args.img

test_img = cv2.imread(test_img_path).astype(np.float32)

# normalize input and convert to torch
mean = np.array([72.39231, 82.908936, 73.1584])
std = np.array([45.31922, 46.152893, 44.914833])

test_img -= mean[np.newaxis, np.newaxis, :]
test_img /= std[np.newaxis, np.newaxis, :]

height, width, _ = test_img.shape
scaled_height = int(height * args.scale_imgs)
scaled_width = int(width *  args.scale_imgs)

test_img = cv2.resize(test_img, (scaled_width, scaled_height))

test_img_torch = torch.from_numpy(test_img).to(device=cuda_device).permute(2, 0, 1).unsqueeze(0) / 255.0

# run model
sem_pred, _, _ = semantic_model.forward(test_img_torch)

# visualize/save result

########## cityscapes visualization from ESP Net: https://github.com/sacmehta/ESPNet ###########################
label_colors = [[128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [0, 0, 0]]

sem_pred_np = sem_pred.squeeze().byte().cpu().data.numpy()
sem_pred_np_color = np.zeros((sem_pred_np.shape[0], sem_pred_np.shape[1], 3), dtype=np.uint8)

for label in range(len(label_colors)):
    sem_pred_np_color[sem_pred_np == label] = label_colors[label]

imsave("data/output/semantic/sem_pred.png", sem_pred_np_color)

# plt.figure()
# plt.imshow(sem_pred_np_color)
# plt.show()

