"""
The main file to run the program.
Please edit the config file in config/config.yaml and run the code.
Made by sayednadim (sayednadim.github.io).
Use it as you wish.
cheers
"""

import os
import csv
from src.dataset.dataset_single_evaluation import build_dataloader_single_evaluation
from src.dataset.dataset_multiple_evaluation import build_dataloader_multiple_evaluation
from src.metrics.image_quality_metrics import ImageQualityMetric
from src.utils.config import get_config
from src.utils.utils import find_folders


def calculate_metrics(config, dataloader, feat_dataloader_img, feat_dataloader_ground_truth):
    metric_class = ImageQualityMetric(config,
                                      dataloader,
                                      feat_dataloader_img,
                                      feat_dataloader_ground_truth)
    metrics = metric_class()
    return metrics


def save_results(config, metrics, name=None):
    if not os.path.exists(config['save_results_path']):
        os.mkdir(config['save_results_path'])
    if name is None:
        save_file_name = config['save_file_name'] + \
                         '_' + config['dataset_name'] + \
                         '_' + config['model_name'] + '.csv'
    else:
        save_file_name = name
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


def main(config):
    """
    Runs the metrics evaluation operations
    and prints the results in the terminal.
    If the save option is set true in the
    config file, then the results will be
    saved in the results directory as a csv
    file.

    For single_evaluation, I assumed the file system like this:

        # ================= structure 1 ===================#

    |- root
    |   |- image_1
    |   |- image_2
    |   | - .....

    For multiple_evaluation, I assumed the file system like this:

    |- root
    |   |- file_10_20
    |        |- image_1
    |        |- image_2
    |        | - .....
    |    |- file_10_20
    |        |- image_1
    |        |- image_2
    |         | - .....

    or like this....

        # ================= structure 2 ===================#

    |- root
    |   |- 01_file
    |       |- file_10_20
    |           |- image_1
    |           |- image_2
    |           | - .....
    |   |- 02_file
    |       |- file_10_20
    |           |- image_1
    |           |- image_2
    |           | - .....


    Please make sure to edit file name for saving....

    args:
    config - path of the config file.
    returns - None.
    """
    # ================= structure 2 ===================#

    if config['multiple_evaluation']:  # check for multiple evaluation
        folder_path = config['generated_image_path']  # select path
        folder_path_list = find_folders(folder_path)  # find the root subfolders
        sub_folder_list = list()  # make a list to hold the sub_folders in the root subfolder
        count = 0  # count number of directories done
        for i in range(len(folder_path_list)):  # start looping over the found subdirs
            sub_folder_list.append(folder_path + '/' + folder_path_list[i])  # add the folders in the sub_folder_list
            individual_folders_list = find_folders(sub_folder_list[i])  # find folders in the first subfolder
            if len(individual_folders_list) != 0:
                for j in range(len(individual_folders_list)):  # go to the first subfolder
                    print("=" * 80)
                    print("Working with structure 2")
                    print("=" * 80)
                    print("=" * 80)
                    print("Currently in {}".format(str(individual_folders_list[i])))
                    print("=" * 80)
                    individual_folders = sub_folder_list[i] + '/' + individual_folders_list[j]  # take the path
                    dataloader = build_dataloader_multiple_evaluation(config=config,
                                                                      data_path=individual_folders,
                                                                      fid=False,
                                                                      fid_ground_truth_or_image='none',
                                                                      batch_size=config['batch_size'],
                                                                      shuffle=False,
                                                                      num_workers=config['threads'])

                    # For FID/IS calculation.
                    feat_dataloader_img = build_dataloader_multiple_evaluation(config=config,
                                                                               data_path=individual_folders,
                                                                               fid=True,
                                                                               fid_ground_truth_or_image='img',
                                                                               batch_size=config['batch_size'],
                                                                               shuffle=False,
                                                                               num_workers=config['threads'])
                    feat_dataloader_ground_truth = build_dataloader_multiple_evaluation(config=config,
                                                                                        data_path=individual_folders,
                                                                                        fid=True,
                                                                                        fid_ground_truth_or_image='ground_truth',
                                                                                        batch_size=config['batch_size'],
                                                                                        shuffle=False,
                                                                                        num_workers=config['threads'])

                    # Calling metric classes
                    metrics = calculate_metrics(config=config, dataloader=dataloader,
                                                feat_dataloader_img=feat_dataloader_img,
                                                feat_dataloader_ground_truth=feat_dataloader_ground_truth)

                    print("Evaluation Results")
                    for key, value in metrics.items():
                        print("{}: {}".format(key, value))
                    print("=" * 80)
                    # saving files
                    if config['save_results']:
                        individual_sub_folder_tail_2 = individual_folders[-10:]  # make sure to edit based on your need!
                        individual_sub_folder_tail_1 = individual_folders[
                                                       -18:-11]  # make sure to edit based on your need!
                        name = config['save_file_name'] + \
                               '_' + config['dataset_name'] + \
                               '_' + config['model_name'] + '_' + \
                               individual_sub_folder_tail_1 + '_' + \
                               individual_sub_folder_tail_2 + '.csv'
                        save_results(config=config, metrics=metrics, name=name)
                    print("Done with {} folder".format(j))
                    print("=" * 80)
                    print("\n")

                    count += 1

            else:
                # ================= structure 1 ===================#
                print("=" * 80)
                print("Working with structure 1")
                print("=" * 80)
                print("=" * 80)
                print("Currently in {}".format(str(sub_folder_list[i])))
                print("=" * 80)
                individual_folders = sub_folder_list[i]
                dataloader = build_dataloader_multiple_evaluation(config=config,
                                                                  data_path=individual_folders,
                                                                  fid=False,
                                                                  fid_ground_truth_or_image='none',
                                                                  batch_size=config['batch_size'],
                                                                  shuffle=False,
                                                                  num_workers=config['threads'])

                # For FID/IS calculation.
                feat_dataloader_img = build_dataloader_multiple_evaluation(config=config,
                                                                           data_path=individual_folders,
                                                                           fid=True,
                                                                           fid_ground_truth_or_image='img',
                                                                           batch_size=config['batch_size'],
                                                                           shuffle=False,
                                                                           num_workers=config['threads'])
                feat_dataloader_ground_truth = build_dataloader_multiple_evaluation(config=config,
                                                                                    data_path=individual_folders,
                                                                                    fid=True,
                                                                                    fid_ground_truth_or_image='ground_truth',
                                                                                    batch_size=config['batch_size'],
                                                                                    shuffle=False,
                                                                                    num_workers=config['threads'])

                # Calling metric classes
                metrics = calculate_metrics(config=config, dataloader=dataloader,
                                            feat_dataloader_img=feat_dataloader_img,
                                            feat_dataloader_ground_truth=feat_dataloader_ground_truth)

                print("Evaluation Results")
                for key, value in metrics.items():
                    print("{}: {}".format(key, value))
                print("=" * 80)
                # saving files
                if config['save_results']:
                    individual_sub_folder_tail_1 = individual_folders[-10:]  # make sure to edit based on your need!
                    name = config['save_file_name'] + \
                           '_' + config['dataset_name'] + \
                           '_' + config['model_name'] + '_' + \
                           individual_sub_folder_tail_1 + '_' + '.csv'
                    save_results(config=config, metrics=metrics, name=name)
                print("=" * 80)
                print("\n")

                count += 1

    else:
        # ================= single evaluation ===================#
        dataloader = build_dataloader_single_evaluation(config=config,
                                                        fid=False,
                                                        fid_ground_truth_or_image='none',
                                                        batch_size=config['batch_size'],
                                                        shuffle=False,
                                                        num_workers=config['threads'])

        # For FID/IS calculation.
        feat_dataloader_img = build_dataloader_single_evaluation(config=config,
                                                                 fid=True,
                                                                 fid_ground_truth_or_image='img',
                                                                 batch_size=config['batch_size'],
                                                                 shuffle=False,
                                                                 num_workers=config['threads'])
        feat_dataloader_ground_truth = build_dataloader_single_evaluation(config=config,
                                                                          fid=True,
                                                                          fid_ground_truth_or_image='ground_truth',
                                                                          batch_size=config['batch_size'],
                                                                          shuffle=False,
                                                                          num_workers=config['threads'])

        # Calling metric classes
        metrics = calculate_metrics(config=config,
                                    dataloader=dataloader,
                                    feat_dataloader_img=feat_dataloader_img,
                                    feat_dataloader_ground_truth=feat_dataloader_ground_truth)

        print("Evaluation Results")
        for key, value in metrics.items():
            print("{}: {}".format(key, value))
        print("=" * 80)
        # saving files
        if config['save_results']:
            save_results(config=config, metrics=metrics)


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
