import torch
from imageio import imread
from skimage.color import rgb2lab
from skimage.transform import rescale
import numpy as np


def get_lvl_name(lvl):
    return '_lvl' + str(lvl)

def load_sample(left_path, right_path):
    # load left/right image
    i0 = imread(left_path)
    if i0.shape[2] == 4:
        i0 = i0[:, :, :3] # remove alpha channel

    i1 = imread(right_path)
    if i1.shape[2] == 4:
        i1 = i1[:, :, :3] # remove alpha channel

    # construct sample
    sample = {'i0': i0, 'i1': i1}

    # apply transforms
    sample['i0_lvl0'] = rgb2lab(sample['i0']).astype('float32')
    sample['i1_lvl0'] = rgb2lab(sample['i1']).astype('float32')
    for lvl in range(2):
        scale = 1.0 / 2**(lvl+1)
        sample['i0_lvl' + str(lvl + 1)] = rgb2lab(rescale(sample['i0'], scale, order=1, anti_aliasing=True,
                                    mode='reflect', multichannel=True)).astype('float32')
        sample['i1_lvl' + str(lvl + 1)] = rgb2lab(rescale(sample['i1'], scale, order=1, anti_aliasing=True,
                                    mode='reflect', multichannel=True)).astype('float32')

    for key in sample.keys():
        sample[key] = torch.from_numpy(sample[key].transpose(2, 0, 1)).unsqueeze(0)

    # construct image pyramid
    I0_pyramid = []
    I1_pyramid = []
    for lvl in range(3):
        I0_pyramid.append(sample['i0' + get_lvl_name(lvl)])
        I1_pyramid.append(sample['i1' + get_lvl_name(lvl)])
    
    return I0_pyramid, I1_pyramid


def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)