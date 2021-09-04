from src.dataset.dataset_single_evaluation import build_dataloader_single_evaluation
from src.dataset.dataset_multiple_evaluation import build_dataloader_multiple_evaluation
from src.evaluation_metrics.image_quality_metrics import calculate_metrics
from src.utils.utils import find_folders
from src.utils.save_utils import save_results


def single_evaluation(config):
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

    args:
    config - path of the config file.
    returns - None.
    """
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
        save_results(config=config, metrics=metrics, name='None')
    print("\n")
    print("=" * 80)
    print("Done with evaluation.")
    print("\n")


def multiple_evaluation(config):
    """
    Runs the metrics evaluation operations
    and prints the results in the terminal.
    If the save option is set true in the
    config file, then the results will be
    saved in the results directory as a csv
    file.

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


    args:
    config - path of the config file.
    returns - None.
    """
    # ================= single evaluation ===================#
    folder_path = config['generated_image_path']  # select path
    folder_path_list = find_folders(folder_path)  # find the root subfolders
    sub_folder_list = list()  # make a list to hold the sub_folders in the root subfolder
    count = 0  # count number of directories done
    for i in range(len(folder_path_list)):  # start looping over the found subdirs
        sub_folder_list.append(folder_path + '/' + folder_path_list[i])  # add the folders in the sub_folder_list
        individual_folders_list = find_folders(sub_folder_list[i])  # find folders in the first subfolder
        if len(individual_folders_list) != 0:
            # ================= structure 2 ===================#
            for j in range(len(individual_folders_list)):  # go to the first subfolder
                print("=" * 80)
                print("Working with structure 2")
                print("=" * 80)
                print("=" * 80)
                print("Currently in {}".format(str(individual_folders_list[j])))
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
                    individual_sub_folder_tail_1 = individual_folders.split('/')[-2]
                    individual_sub_folder_tail_2 = individual_folders.split('/')[-1]
                    name =  individual_sub_folder_tail_1 + '/' + \
                           individual_sub_folder_tail_2
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
                individual_sub_folder_tail_1 = individual_folders.split('/')[-1]  # make sure to edit based on your need!
                print(individual_folders)
                name = individual_sub_folder_tail_1
                save_results(config=config, metrics=metrics, name=name)
            print("=" * 80)
            print("\n")

            count += 1
    print("\n")
    print("=" * 80)
    print("Done with evaluation.")
    print("\n")
