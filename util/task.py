import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random

###################################################################
# random mask generation
###################################################################


def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    mask[:, x:range_x, y:range_y] = 0

    return mask


def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 64)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask

###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs