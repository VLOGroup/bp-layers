import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import correlate

def makeColorWheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    size = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((3, size))

    col = 0
    # RY
    colorwheel[0, col:col+RY] = 255
    colorwheel[1, col:col+RY] = np.floor(255 * np.arange(RY)/RY)
    col += RY

    # YG
    colorwheel[0, col:col+YG] = 255 - np.floor(255 * np.arange(YG)/YG)
    colorwheel[1, col:col+YG] = 255
    col += YG

    # GC
    colorwheel[1, col:col+GC] = 255
    colorwheel[2, col:col+GC] = np.floor(255 * np.arange(GC)/GC)
    col += GC

    # CB
    colorwheel[1, col:col+CB] = 255 - np.floor(255 * np.arange(CB)/CB)
    colorwheel[2, col:col+CB] = 255
    col += CB

    # BM
    colorwheel[0, col:col+BM] = np.floor(255 * np.arange(BM)/BM)
    colorwheel[2, col:col+BM] = 255
    col += BM

    # MR
    colorwheel[0, col:col+MR] = 255
    colorwheel[2, col:col+MR] = 255 - np.floor(255 * np.arange(MR)/MR)

    return colorwheel.astype('uint8')

def computeNormalizedFlow(u, v, u_ref=None, v_ref=None, verbose=False):
    # copy to not overwrite the inputs 
    u = u.copy()
    v = v.copy()

    eps = 1e-15
    UNKNOWN_FLOW_THRES = 1e9
    # UNKNOWN_FLOW = 1e10

    maxu = -999
    maxv = -999
    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = np.logical_or(np.abs(u) > UNKNOWN_FLOW_THRES, np.abs(v) > UNKNOWN_FLOW_THRES)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = np.maximum(maxu, np.max(u))
    minu = np.minimum(minu, np.min(u))

    maxv = np.maximum(maxv, np.max(v))
    minv = np.minimum(minv, np.min(v))
    
    if u_ref is not None and v_ref is not None:
        rad = np.sqrt(u_ref**2 + v_ref**2)
    else:
        rad = np.sqrt(u**2 + v**2)
    maxrad = np.maximum(maxrad, np.max(rad))

    if verbose:
        print("max flow: ", maxrad, " flow range: u = ", minu, "..", maxu, "v = ", minv, "..", maxv)

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    return u, v

def computeFlowImg(u, v, u_ref=None, v_ref=None):
    # do not overwrite input flow!
    u = u.copy()
    v = v.copy()

    u, v = computeNormalizedFlow(u, v, u_ref, v_ref)

    nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
    u[nanIdx] = 0
    v[nanIdx] = 0

    cw = makeColorWheel().T

    M, N = u.shape
    img = np.zeros((M, N, 3)).astype('uint8')

    mag = np.sqrt(u**2 + v**2)

    phi = np.arctan2(-v, -u) / np.pi # [-1, 1]
    phi_idx = (phi + 1.0) / 2.0 * (cw.shape[0] - 1)
    f_phi_idx = np.floor(phi_idx).astype('int')

    c_phi_idx = f_phi_idx + 1
    c_phi_idx[c_phi_idx == cw.shape[0]] = 0

    floor = phi_idx - f_phi_idx

    for i in range(cw.shape[1]):
        tmp = cw[:, i]

        # linear blend between colors
        col0 = tmp[f_phi_idx] / 255.0 # from colorwheel take specified values in phi_idx
        col1 = tmp[c_phi_idx] / 255.0
        col = (1.0 - floor)*col0 + floor * col1

        # increase saturation for small magnitude
        sat_idx = mag <= 1
        col[sat_idx] = 1 - mag[sat_idx] * (1 - col[sat_idx])

        col[np.logical_not(sat_idx)] = col[np.logical_not(sat_idx)] * 0.75

        img[:, :, i] = (np.floor(255.0*col*(1-nanIdx))).astype('uint8')
    return img

