import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.SFUNet_arch import SFUNet
from skimage import img_as_ubyte
from skimage import io
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Real Image Denoising')

parser.add_argument('--input_dir', default='datasets/test/SIDD/input_crops/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/xformer_real_dn_sidd/', type=str, help='Directory for results')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--weights', default='experiments/pretrained_models/SFUNet_real_dn.pth', type=str, help='Path to model weights')

args = parser.parse_args()

####### Load model options #######
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt_str = r"""
  type: SFUNet
  inp_channels: 3
  out_channels: 3
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
  LayerNorm_type: WithBias
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

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)
    
    # Create subdirectories for different image types
    noisy_dir = os.path.join(result_dir_png, 'noisy')
    denoised_dir = os.path.join(result_dir_png, 'denoised')
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)

model_restoration = SFUNet(**opt)

weights = args.weights
checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data - load PNG images from input_crops directory
files = natsorted(glob(os.path.join(args.input_dir, '*.png')))
print(f"Found {len(files)} images in {args.input_dir}")

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        # Load noisy image
        img = np.float32(utils.load_img(file_))/255.
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()
        
        # Padding for wavelet transform (ensure dimensions are multiples of 16)
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        H = ((h+15)//16)*16
        W = ((w+15)//16)*16
        padh = H-h
        padw = W-w
        img_tensor = F.pad(img_tensor, (0,padw,0,padh), 'reflect')
        
        # Denoise
        restored_tensor = model_restoration(img_tensor)
        
        # Unpad to original dimensions
        restored_tensor = restored_tensor[:,:,:h,:w]
        
        # Convert back to numpy
        restored_img = torch.clamp(restored_tensor,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        if args.save_images:
            # Get filename without path
            filename = os.path.split(file_)[-1]
            
            # Save noisy image
            noisy_save_file = os.path.join(noisy_dir, filename)
            utils.save_img(noisy_save_file, img_as_ubyte(img))
            
            # Save denoised image
            denoised_save_file = os.path.join(denoised_dir, filename)
            utils.save_img(denoised_save_file, img_as_ubyte(restored_img))
