# Image Quality Evaluation Metrics (On-going)
Implementation of Common Image Evaluation Metrics by [Sayed Nadim](sayednadim.github.io). The repo is built based on full reference image quality metrics such as L1, L2, PSNR, SSIM, LPIPS. and feature-level quality metrics such as FID, IS. It can be used for evaluating image denoising, colorization, inpainting, deraining, dehazing etc. where we have access to ground truth.

The goal of this repo is to provide a common evaluation script for image evaluation tasks. It contains some commonly used image quality metrics for image evaluation (e.g., L1, L2, SSIM, PSNR, [LPIPS](https://github.com/richzhang/PerceptualSimilarity), FID, IS). 

Pull requests and  corrections/suggestions will be cordially appreciated. 

### Please Note
- Images are scaled to [0,1]. If you need to change the data range, please make sure to change the data range in SSIM and PSNR.
- Number of generated images and ground truth images have to be exactly same. 
- I have resized the images to be (`256,256`). You can change the resolution based on your needs.
- Please make sure that all the images (generated and gt images) are in the corresponding folders. 

### Requirements
- PyTorch ( `>= 1.3` )
- Python ( `>=3.5` )
- [PyTorch Image Quality (PIQ)](https://github.com/photosynthesis-team/piq) ( `$ pip install piq` )


### How to use
- Edit [`config.yaml`](https://github.com/SayedNadim/Image-Quality-Evaluation-Metrics/blob/master/src/config/config.yaml) as per your need.
- Run `main.py`


### Usage
- Options in `config.yaml` file
  - `dataset_name` - Name of the dataset (e.g. Places, DIV2K etc. Used for saving dataset name in csv file.). Default - Places
  - `dataset_with_subfolders` - Set to `True` if your dataset has sub-folders containing images. Default - False
  - `dataset_format` - Whether you are providing flists or just path to the image folders. Default - image.
  - `model_name1` - Name of the model. Used for saving metrics values in the CSV. Default - Own.
  - `generated_image_path` - Path to your generated images.
  - `gt_image_path` - Path to your ground truth images.
  - `batch_size` - batch size you want to use. Default - 4.
  - `image_shape` - Shape of the image. Both generated image and ground truth images will be resized to this width. Default -  [256, 256, 3].
  - `threads` - Threads to be used for multi-processing Default - 4.
  - `random_crop` - If you want random cropped image, instead of resized. Currently not implemented.
  - `save_results` - If you want to save the results in csv files. Saved to `results` folder. Default - True.
  - `save_type` - csv or npz. npz is not implemented yet.

### To-do
- [x] L1
- [x] L2
- [x] SSIM
- [x] PSNR
- [x] LPIPS
- [x] FID
- [x] IS

### Acknowledgement
Thanks to [PhotoSynthesis Team](https://github.com/photosynthesis-team/piq) for the wonderful implementation of the metrics. Please cite accordingly if you use PIQ for the evaluation.

Cheers!!
