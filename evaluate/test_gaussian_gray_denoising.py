import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.SFUNet_arch import SFUNet
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Gasussian Grayscale Denoising')

parser.add_argument('--input_dir', default='datasets/DFWB/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=None, type=str, help='Directory for results (auto-generated if not specified)')
parser.add_argument('--sigma', default='15', type=str, help='Sigma values, 15, 25, or 50')

args = parser.parse_args()

####### Load model options #######
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt_str = r"""
  type: SFUNet
  inp_channels: 1
  out_channels: 1
  img_size: 128
  dim: 48
  num_blocks: [2, 4, 4]
  spatial_num_blocks: [2,4,4,6]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  window_size: [16,16,16,16]
  drop_path_rate: 0.1
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: BiasFree
  dual_pixel_task: False
  wavelet_type: db4
  use_memory_efficient: true
  use_lightweight_gate: true
  use_adaptive_wavelet: false
  wavelet_types: ['db4']
"""
opt = yaml.safe_load(opt_str)
network_type = opt.pop('type')
##########################################

sigma = np.int_(args.sigma)

if args.result_dir is None:
    args.result_dir = f'./results/SFUNet_gray_dn_sigma{sigma}/'

factor = 8

datasets = ['Set12', 'BSD68', 'Urban100']

print("Compute results for noise level",sigma)
model_restoration = SFUNet(**opt)    

if sigma == 15:
    weights = 'experiments/GaussianGrayDenoising_SFUNet_Sigma15/models/net_g_400000.pth'
elif sigma == 25:
    weights = 'experiments/GaussianGrayDenoising_SFUNet_Sigma25/models/net_g_400000.pth'
else:
    weights = 'experiments/GaussianGrayDenoising_SFUNet_Sigma50/models/net_g_400000.pth'

checkpoint = torch.load(weights, map_location='cpu')
if 'params' in checkpoint:
    model_restoration.load_state_dict(checkpoint['params'])
else:
    model_restoration.load_state_dict(checkpoint)


print("===>Testing using weights: ",weights)
print("------------------------------------------------")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

for dataset in datasets:
    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
    result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma))
    os.makedirs(result_dir_tmp, exist_ok=True)
    noisy_dir = os.path.join(result_dir_tmp, 'noisy')
    os.makedirs(noisy_dir, exist_ok=True)

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(utils.load_gray_img(file_))/255.

            np.random.seed(seed=0)  # for reproducibility
            img += np.random.normal(0, sigma/255., img.shape)

            noisy_save_file = os.path.join(noisy_dir, os.path.split(file_)[-1])
            utils.save_gray_img(noisy_save_file, img_as_ubyte(np.clip(img, 0, 1)))

            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()


            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H = ((h+15)//16)*16
            W = ((w+15)//16)*16
            padh = H-h
            padw = W-w
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
            utils.save_gray_img(save_file, img_as_ubyte(restored))
