import numpy as np
import argparse
from PIL import Image
import math
from dataloader.image_folder import make_dataset
import os
import glob
import shutil

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--gt_path', type = str, default='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/image_painting/imagenet_test.txt',
                    help = 'path to original particular solutions')
parser.add_argument('--save_path', type = str, default='/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/image_painting/results/ours/imagenet/center',
                    help='path to save the test dataset')
parser.add_argument('--num_test', type=int, default=1000,
                    help='how many images to load for each test')
# parser.add_argument('--sample_numbers',type=int,default=50, help='how many smaple images for testing')
args = parser.parse_args()


def compute_errors(gt, pre):

    # l1 loss
    l1 = np.mean(np.abs(gt-pre))

    # PSNR
    mse = np.mean((gt - pre) ** 2)
    if mse == 0:
        PSNR = 100
    else:
        PSNR = 20 * math.log10(255.0 / math.sqrt(mse))

    # TV
    gx = pre - np.roll(pre, -1, axis=1)
    gy = pre - np.roll(pre, -1, axis=0)
    grad_norm2 = gx ** 2 + gy ** 2
    TV = np.mean(np.sqrt(grad_norm2))

    return l1, PSNR, TV


if __name__ == "__main__":

    gt_paths, gt_size = make_dataset(args.gt_path)
    #pre_paths, pre_size = make_dataset(args.save_path)

    # for i in range(20000):
    #     print(i)
    #     name = gt_paths[i].split("/")[-1]
    #     path = os.path.join(args.save_path,name)
    #     try:
    #         image = Image.open(path)
    #     except:
    #         print (path)

    # assert gt_size == pre_size
    #
    iters = int(20000/args.num_test)

    l1_loss = np.zeros(iters, np.float32)
    PSNR = np.zeros(iters, np.float32)
    TV = np.zeros(iters, np.float32)

    for i in range(0, iters):
        l1_iter = np.zeros(args.num_test, np.float32)
        PSNR_iter = np.zeros(args.num_test, np.float32)
        TV_iter = np.zeros(args.num_test, np.float32)

        num = i*args.num_test

        for j in range(args.num_test):
            index = num+j
            gt_image = Image.open(gt_paths[index]).resize([256,256]).convert('RGB')
            gt_numpy = np.array(gt_image).astype(np.float32)
            l1_sample = 1000
            PSNR_sample = 0
            TV_sample = 1000
            name = gt_paths[index].split('/')[-1].split(".")[0]+"*"
            pre_paths = sorted(glob.glob(os.path.join(args.save_path, name)))
            num_image_files = len(pre_paths)

            for k in range(num_image_files-1):
                index2 = k
                try:
                    pre_image = Image.open(pre_paths[index2]).resize([256,256]).convert('RGB')
                    pre_numpy = np.array(pre_image).astype(np.float32)
                    l1_temp, PSNR_temp, TV_temp = compute_errors(gt_numpy, pre_numpy)
                    # select the best results for the errors estimation
                    if l1_temp - PSNR_temp + TV_temp < l1_sample - PSNR_sample + TV_sample:
                        l1_sample, PSNR_sample, TV_sample = l1_temp, PSNR_temp, TV_temp
                        best_index = index2
                except:
                    print(pre_paths[index2])
            shutil.copy(pre_paths[best_index], '/media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b2557/dataset/image_painting/results/ours/imagenet/center_copy/')
            print(pre_paths[best_index])
            print(l1_sample, PSNR_sample, TV_sample)

            l1_iter[j], PSNR_iter[j], TV_iter[j] = l1_sample, PSNR_sample, TV_sample

        l1_loss[i] = np.mean(l1_iter)
        PSNR[i] = np.mean(PSNR_iter)
        TV[i] = np.mean(TV_iter)

        print(i)
        print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss[i], PSNR[i], TV[i]))

    print('{:>10},{:>10},{:>10}'.format('L1_LOSS', 'PSNR', 'TV'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss.mean(), PSNR.mean(), TV.mean()))
