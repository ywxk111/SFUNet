# SFUNet: Gated Spatial-Frequency Fusion for Enhanced Image Denoising
<img width="415" height="206" alt="image" src="https://github.com/user-attachments/assets/dd2c7f2f-3e42-4433-9e79-76aec89af383" />

This is the official implementation of the paper:
**Gated Spatial-Frequency Fusion for Enhanced Image Denoising**

---

## Important Note
This code is **directly associated with the manuscript submitted to The Visual Computer**.
If you use this code in your research, please cite our corresponding paper.

---

## Requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- opencv-python
- Pillow
- wavelet

You can install dependencies via:pip install -r requirements.txt

## Training

  ## Train on GaussionColor image denoising

python -m torch.distributed.launch --nproc_per_node=4 --master_port=2418 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma15.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2416 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma25.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/GaussianColorDenoising_X_FormerSigma50.yml --launcher pytorch

  ## Train on GaussionGrayscale image denoising

python -m torch.distributed.launch --nproc_per_node=4 --master_port=2418 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma15.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2416 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma25.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2414 basicsr/train.py -opt options/GaussianGrayscaleDenoising_X_FormerSigma50.yml --launcher pytorch


  ## Train on Real Image Denoising
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6414 basicsr/train.py -opt options/RealDenoising_X_Former.yml --launcher pytorch


## Test
  ## Test on Gaussian Color Image Denoising
  noise 15
python evaluate/test_gaussian_color_denoising.py --sigma 15
python evaluate/evaluate_gaussian_color_denoising.py --sigma 15
  noise 25
python evaluate/test_gaussian_color_denoising.py --sigma 25
python evaluate/evaluate_gaussian_color_denoising.py --sigma 25
  noise 50
python evaluate/test_gaussian_color_denoising.py --sigma 50
python evaluate/evaluate_gaussian_color_denoising.py --sigma 50

  ## Test on Gaussian Grayscale Image Denoising
  noise 15
python evaluate/test_gaussian_gray_denoising.py --sigma 15
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 15
  noise 25
python evaluate/test_gaussian_gray_denoising.py --sigma 25
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 25
  noise 50
python evaluate/test_gaussian_gray_denoising.py --sigma 50
python evaluate/evaluate_gaussian_gray_denoising.py --sigma 50

