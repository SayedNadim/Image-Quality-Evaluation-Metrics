import os
from src.dataset.dataset import build_dataloader
from src.metrics.image_quality_metrics import Image_Quality_Metric
from src.utils.config import Config
import csv


def main(config):
    dataloader = build_dataloader(config=config, fid=False, fid_gt_or_image='none', batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.threads)

    # For FID/IS calculation.
    feat_dataloader_img = build_dataloader(config=config, fid=True, fid_gt_or_image='img', batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.threads)
    feat_dataloader_gt = build_dataloader(config=config, fid=True, fid_gt_or_image='gt', batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.threads)

    # Calling metric classes
    metric_class = Image_Quality_Metric(config, dataloader, feat_dataloader_img, feat_dataloader_gt)
    metrics = metric_class()

    print("Evaluation Results")
    for key, value in metrics.items():
        print("{}: {}\n".format(key, value))

    # saving files
    if config.save_results:
        if not os.path.exists(config.save_results_path):
            os.mkdir(config.save_results_path)
        save_file_name = config.save_file_name + '_' + config.dataset_name + '_' + config.model_name + '.csv'
        if not os.path.exists(os.path.join(config.save_results_path, save_file_name)):
            os.mknod(os.path.join(config.save_results_path, save_file_name))
        result_file_path = os.path.join(config.save_results_path, save_file_name)

        with open(result_file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in metrics.items():
                writer.writerow([key, value.data.item()])

        print("\n"
              "Saved metric in {}."
              "\n".format(config.save_results_path))

        # with open(result_file_path) as csv_file:
        #     reader = csv.reader(csv_file)
        #     mydict = dict(reader)


if __name__ == '__main__':
    print(
        "Implementation of Common Image Evaluation Metrics "
        "by Sayed Nadim (sayednadim.github.io)"
        "\n"
        "The repo is built based on full reference image quality metrics such as L1, L2, PSNR, SSIM, LPIPS."
        "\n"
        "and feature-level quality metrics such as FID, IS."
        "\n"
        "It can be used for evaluating image denoising, colorization, inpainting, deraining, dehazing etc. "
        "where we have access to ground truth."
        "\n"
    )
    print("-" * 80)
    config_path = 'src/config/config.yaml'
    config_file = Config(config_path)
    main(config_file)
