"""
The class file containing all the quality metrics and operations.
"""

import time
import warnings

import torch
from torch import nn

import piq

from tqdm import tqdm

from src.evaluation_metrics.psnr import psnr_calculation

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_metrics(config, dataloader, feat_dataloader_img, feat_dataloader_ground_truth):
    metric_class = ImageQualityMetric(config,
                                      dataloader,
                                      feat_dataloader_img,
                                      feat_dataloader_ground_truth)
    metrics = metric_class()
    return metrics



class ImageQualityMetric(nn.Module):
    """
    Image quality metric class.
    This will calculate the l1,
    l2, ssim, psnr, lpips, fid,
    is scores and return to the
    main function.
    """

    def __init__(self, config, dataloader, feature_dataloader_img, feature_dataloader_ground_truth):
        super().__init__()
        self.config = config
        self.dataloader = dataloader
        self.feature_dataloader_img = feature_dataloader_img
        self.feature_dataloader_ground_truth = feature_dataloader_ground_truth
        self.ssim_distance = piq.SSIMLoss()
        self.l1_distance = nn.L1Loss()
        self.lpips_class = piq.LPIPS(distance='mae')
        self.fid_class = piq.FID()
        self.is_class = piq.IS()
        self.gs_class = piq.GS()
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.l1_distance.cuda()
            self.ssim_distance.cuda()
            self.fid_class.cuda()
            self.is_class.cuda()
            self.gs_class.cuda()
        else:
            self.device = 'cpu'

        if config['show_config']:
            print("Configuration")
            print("=" * 80)
            for key, value in config.items():
                print("{}: {}".format(key, value))
            print("=" * 80)
        print("Dataset Statistics - "
              "Total {} images found.".format(len(self.dataloader) * self.config['batch_size']))
        print("=" * 80)
        print("Calculating image-level inpainting metrics. "
              "Please wait till the progress bar in 100%."
              )
        print("=" * 80)

    def forward(self):
        """
        Image-level metrics are calculated batch-wise
        while the feature-level metrics are calculated
        as whole dataset.
        """
        l1_value, l2_value, ssim_value, psnr_value, lpips_value = [], [], [], [], []

        t_0 = time.time()

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                time.sleep(0.1)
                image, ground_truth = batch["images"], batch["ground_truth"]
                image = image.to(self.device)
                ground_truth = ground_truth.to(self.device)

                psnr, mse = psnr_calculation(image, ground_truth, data_range=1.)
                l2_value.append(mse)
                psnr_value.append(psnr)
                l1_value.append(self.l1_distance(image, ground_truth))
                ssim_value.append((1. - self.ssim_distance(image, ground_truth)))
                lpips_value.append(self.lpips_class(image, ground_truth))

            l1_mean = torch.mean(torch.stack(l1_value))  # pylint: disable=maybe-no-member
            l2_mean = torch.mean(torch.stack(l2_value))  # pylint: disable=maybe-no-member
            ssim_mean = torch.mean(torch.stack(ssim_value))  # pylint: disable=maybe-no-member
            psnr_mean = torch.mean(torch.stack(psnr_value))  # pylint: disable=maybe-no-member
            lpips_mean = torch.mean(torch.stack(lpips_value))  # pylint: disable=maybe-no-member

            # # fid
            print("=" * 80)
            print("Computing feature-level inpainting metrics. "
                  "Please wait. It may take some moment."
                  )
            print("=" * 80)
            img_load, ground_truth_load = \
                self.feature_dataloader_img, self.feature_dataloader_ground_truth
            img_feature = self.fid_class.compute_feats(img_load)
            ground_truth_feature = self.fid_class.compute_feats(ground_truth_load)
            fid_value = self.fid_class.compute_metric(img_feature, ground_truth_feature)
            t_2 = time.time()

            print("=" * 80)
            print(
                "\n"
                "Finished in {:4f}s."
                "\n".format(t_2 - t_0))
            print("=" * 80)

            return {"l1": l1_mean,
                    "l2": l2_mean,
                    "ssim": ssim_mean,
                    "psnr": psnr_mean,
                    "lpips": lpips_mean,
                    "fid": fid_value}
