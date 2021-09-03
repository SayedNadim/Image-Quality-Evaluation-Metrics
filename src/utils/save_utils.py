import os
import csv


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
