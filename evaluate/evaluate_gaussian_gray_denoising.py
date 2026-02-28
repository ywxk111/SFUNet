import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import csv
import utils

def proc(filename):
    tar,prd = filename
    tar_img = utils.load_gray_img(tar)
    prd_img = utils.load_gray_img(prd)
        
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM

parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising')

parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

sigmas = np.int_(args.sigmas.split(','))

datasets = ['Set12', 'BSD68', 'Urban100']

for dataset in datasets:

    gt_path = os.path.join('datasets','DFWB', 'test', dataset)
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
    assert len(gt_list) != 0, "Target files not found"

    for sigma_test in sigmas:
        file_path = os.path.join('results', 'SFUNet_gray_dn_sigma' + str(sigma_test), dataset, str(sigma_test))
        path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
        assert len(path_list) != 0, "Predicted files not found"

        psnr, ssim = [], []
        img_files =[(i, j) for i,j in zip(gt_list,path_list)]
        
        csv_path = os.path.join(file_path, f'{dataset}_sigma{sigma_test}_metrics.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        print(f'\n=== {dataset} Dataset (Sigma={sigma_test}) ===')
        print(f'{"Filename":<25} {"PSNR":<8} {"SSIM":<8}')
        print('-' * 45)
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'psnr', 'ssim'])
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                for (gt_fp, prd_fp), PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
                    ps, ss = PSNR_SSIM[0], PSNR_SSIM[1]
                    psnr.append(ps)
                    ssim.append(ss)
                    fname = os.path.basename(prd_fp)
                    print(f'{fname:<25} {ps:<8.4f} {ss:<8.4f}')
                    writer.writerow([fname, f'{ps:.6f}', f'{ss:.6f}'])

        avg_psnr = sum(psnr)/len(psnr)
        avg_ssim = sum(ssim)/len(ssim)

        print('-' * 45)
        print(f'{"Average":<25} {avg_psnr:<8.4f} {avg_ssim:<8.4f}')
        print(f'CSV saved to: {csv_path}')
        print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
