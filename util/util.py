import numpy as np
import os
import imageio
import math
import torch


# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 2:
            image_numpy = flow_to_image(image_numpy)
            return image_numpy.astype(imtype)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


# conver a tensor into a numpy array
def tensor2array(value_tensor):
    if value_tensor.dim() == 3:
        numpy = value_tensor.view(-1).cpu().float().numpy()
    else:
        numpy = value_tensor[0].view(-1).cpu().float().numpy()
    return numpy


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    maxrad = -1
    u = flow[0, :, :]
    v = flow[1, :, :]
    idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxunknow] = 0
    v[idxunknow] = 0
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(maxrad, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    return img


def show_attention(attention, position):
    number = position[0] * position[1]
    if number > attention.size(1):
        raise NotImplementedError('query position overflow the attention map')
    attention_map = attention[:,number]
    value = torch.topk(attention_map, 80)
    for i in range(80):
        attention_map[value[1][i]] = 1 - (1-0.0)/40*i
    w = int(math.sqrt(attention.size(1)))
    attention_map = attention_map.view(1, w, w)
    attention_map = torch.where(attention_map>value[0][-1], attention_map, torch.zeros_like(attention_map))
    # attention_map = ((attention_map-attention_map.min())/(attention_map.max()-attention_map.min())-0.5)*2
    # attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    # attention_map[:, position[0], position[1]] = 1

    # attention weight
    attention_weight = attention.sum(dim=0).view(1, w, w)
    attention_weight = ((attention_weight-attention_weight.min())/(attention_weight.max()-attention_weight.min())-0.5)*2

    return  attention_map, attention_weight


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
