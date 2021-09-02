"""
The main file to run the program.
Please edit the config file in config/config.yaml and run the code.
Made by sayednadim (sayednadim.github.io).
Use it as you wish.
cheers
"""

import os
import csv
from src.dataset.dataset import build_dataloader
from src.metrics.image_quality_metrics import ImageQualityMetric
from src.utils.config import get_config


def main(config):
    """
    Runs the metrics evaluation operations
    and prints the results in the terminal.
    If the save option is set true in the
    config file, then the results will be
    saved in the results directory as a csv
    file.

    args:
    config - path of the config file.
    returns - None.
    """
    dataloader = build_dataloader(config=config,
                                  fid=False,
                                  fid_ground_truth_or_image='none',
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=config['threads'])

    # For FID/IS calculation.
    feat_dataloader_img = build_dataloader(config=config,
                                           fid=True,
                                           fid_ground_truth_or_image='img',
                                           batch_size=config['batch_size'],
                                           shuffle=False,
                                           num_workers=config['threads'])
    feat_dataloader_ground_truth = build_dataloader(config=config,
                                                    fid=True,
                                                    fid_ground_truth_or_image='ground_truth',
                                                    batch_size=config['batch_size'],
                                                    shuffle=False,
                                                    num_workers=config['threads'])

    # Calling metric classes
    metric_class = ImageQualityMetric(config, dataloader, feat_dataloader_img, feat_dataloader_ground_truth)
    metrics = metric_class()

    print("Evaluation Results")
    for key, value in metrics.items():
        print("{}: {}".format(key, value))
    print("-" * 80)
    # saving files
    if config['save_results']:
        if not os.path.exists(config['save_results_path']):
            os.mkdir(config['save_results_path'])
        save_file_name = config['save_file_name'] + \
                         '_' + config['dataset_name'] + \
                         '_' + config['model_name'] + '.csv'
        if not os.path.exists(os.path.join(config['save_results_path'], save_file_name)):
            os.mknod(os.path.join(config['save_results_path'], save_file_name))
        result_file_path = os.path.join(config['save_results_path'], save_file_name)

        with open(result_file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in metrics.items():
                writer.writerow([key, value.data.item()])

        print("\n"
              "Saved metric in {}."
              "\n".format(config['save_results_path']))

        # with open(result_file_path) as csv_file:
        #     reader = csv.reader(csv_file)
        #     mydict = dict(reader)


if __name__ == '__main__':
    print(
        "#====================================================================================================#"
        "\n"
        "#                                Common Image Evaluation Metrics                                     #"
        "\n"
        "#====================================================================================================#"
        "\n"
        "# Implementation of Common Image Evaluation Metrics by Sayed Nadim (sayednadim.github.io)            #"
        "\n"
        "# The repo is built based on full reference image quality metrics such as L1, L2, PSNR, SSIM, LPIPS. #"
        "\n"
        "# and feature-level quality metrics such as FID, IS. It can be used for evaluating image denoising,  #"
        "\n"
        "# colorization, inpainting, deraining, dehazing etc. supervised tasks.                               #"
        "\n"
        "#====================================================================================================#"
    )
    CONFIG_PATH = 'src/config/config.yaml'
    CONFIG_FILE = get_config(CONFIG_PATH)
    main(CONFIG_FILE)
