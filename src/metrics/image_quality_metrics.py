import torch
import torch.nn as nn

import piq

from tqdm import tqdm
import time
import warnings

from src.metrics.psnr import psnr_calculation

# from src.metrics.ssim import ssim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

'''
to suppress this warning from PIQ
/home/la-belva/anaconda3/envs/latest/lib/python3.8/site-packages/piq/perceptual.py:148: 
UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() 
or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
self.weights = [torch.tensor(w) for w in weights]
'''


class Image_Quality_Metric(nn.Module):
    def __init__(self, config, dataloader, feature_dataloader_img, feature_dataloader_gt):
        super().__init__()
        self.config = config
        self.dataloader = dataloader
        self.feature_dataloader_img = feature_dataloader_img
        self.feature_dataloader_gt = feature_dataloader_gt
        self.l1_distance = nn.L1Loss()
        self.lpips = piq.LPIPS(distance='mae')
        self.ssim_distance = piq.SSIMLoss()
        self.fid_class = piq.FID()
        self.is_class = piq.IS()
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.lpips.cuda()
            self.fid_class.cuda()
            self.is_class.cuda()
        else:
            self.device = 'cpu'


        print("\n"
              "Dataset Statistics - "
              "Total {} images found.".format(len(self.dataloader) * self.config.batch_size))
        print("-" * 80)
        print("Calculating image-level inpainting metrics. Please wait till the progress bar in 100%."
              )
        print("-" * 80)

    def __call__(self):
        l1_value = []
        l2_value = []
        ssim_value = []
        psnr_value = []
        lpips_value = []

        t0 = time.time()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader)):
                time.sleep(0.1)
                image, gt = batch["images"], batch["gt"]
                image = image.to(self.device)
                gt = gt.to(self.device)

                psnr, mse = psnr_calculation(image, gt, data_range=1.)
                l2_value.append(mse)
                psnr_value.append(psnr)
                l1_value.append(self.l1_distance(image, gt))
                ssim_value.append((1. - self.ssim_distance(image, gt)))
                lpips_value.append(self.lpips(image, gt))

            l1_mean = torch.mean(torch.stack(l1_value))
            l2_mean = torch.mean(torch.stack(l2_value))
            ssim_mean = torch.mean(torch.stack(ssim_value))
            psnr_mean = torch.mean(torch.stack(psnr_value))
            lpips_mean = torch.mean(torch.stack(lpips_value))

            # # fid
            print("-" * 80)
            print("Computing feature-level inpainting metrics. Please wait. It may take some moment."
                  )
            print("-" * 80)
            img_load, gt_load = self.feature_dataloader_img, self.feature_dataloader_gt
            img_feature = self.fid_class.compute_feats(img_load)
            gt_feature = self.fid_class.compute_feats(gt_load)
            fid_value = self.fid_class.compute_metric(img_feature, gt_feature)
            img_feature_is = self.is_class.compute_feats(img_load)
            gt_feature_is = self.is_class.compute_feats(gt_load)
            is_value = self.is_class.compute_metric(img_feature_is, gt_feature_is)

            t2 = time.time()

            print("-" * 80)
            print(
                "\n"
                "Finished in {:4f}s."
                "\n".format(t2 - t0))
            print("-" * 80)

            return {"l1": l1_mean, "l2": l2_mean, "ssim": ssim_mean, "psnr": psnr_mean, "lpips": lpips_mean,
                    "fid": fid_value, "is": is_value}
