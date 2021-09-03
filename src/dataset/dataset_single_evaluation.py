import os
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.utils import natural_keys, allowed_image_extensions, image_reader, find_samples_in_subfolders, \
    default_flist_reader

from PIL import Image


class DatasetSingleEvaluation(data.Dataset):
    def __init__(self, config, fid=False, fid_ground_truth_or_image='none'):
        super(DatasetSingleEvaluation, self).__init__()
        self.config = config
        self.fid = fid
        self.fid_ground_truth_or_img = fid_ground_truth_or_image
        if config['dataset_format'] == 'image':
            if config['dataset_with_subfolders']:
                self.ground_truth_samples = find_samples_in_subfolders(config['ground_truth_image_path'])
                self.img_samples = find_samples_in_subfolders(config['generated_image_path'])
            else:
                self.ground_truth_samples = [os.path.join(config['ground_truth_image_path'], x) for x in
                                             os.listdir(config['ground_truth_image_path']) if
                                             allowed_image_extensions(x)]
                self.img_samples = [os.path.join(config['generated_image_path'], x) for x in
                                    os.listdir(config['generated_image_path']) if allowed_image_extensions(x)]
        elif config['dataset_format'] == 'file_list':
            self.ground_truth_samples = default_flist_reader(config['ground_truth_image_path'])
            self.img_samples = default_flist_reader(config['generated_image_path'])

        self.ground_truth_samples.sort(key=natural_keys)
        self.img_samples.sort(key=natural_keys)
        self.image_shape = config['image_shape'][:2]
        self.dataset_name = config['dataset_name']
        self.return_dataset_name = config['return_dataset_name']

        if config['random_crop']:
            Warning("Random crop is not implemented yet. Images are being resized.")

    def __getitem__(self, index):
        img = image_reader(self.img_samples[index])
        ground_truth = image_reader(self.ground_truth_samples[index])

        img = Image.fromarray(img)
        ground_truth = Image.fromarray(ground_truth)

        img = transforms.Resize(self.image_shape)(img)
        ground_truth = transforms.Resize(self.image_shape)(ground_truth)

        img = transforms.ToTensor()(img)
        ground_truth = transforms.ToTensor()(ground_truth)

        if self.fid:
            if self.fid_ground_truth_or_img == 'img':
                return {"images": img}
            elif self.fid_ground_truth_or_img == 'ground_truth':
                return {"images": img}
            else:
                raise KeyError(
                    "FID/IS is true but return type is none. "
                    "Please make two dataloaders and select img/ground_truth as "
                    "inputs for the dataloaders")
        else:
            if self.return_dataset_name:
                return {"images": img, "ground_truth": ground_truth, "name": self.dataset_name}
            else:
                return {"images": img, "ground_truth": ground_truth}

    def __len__(self):
        return len(self.ground_truth_samples)


def build_dataloader_single_evaluation(config, fid, fid_ground_truth_or_image, batch_size,
                     num_workers, shuffle=False):
    dataset = DatasetSingleEvaluation(
        config=config,
        fid=fid,
        fid_ground_truth_or_image=fid_ground_truth_or_image
    )

    # print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=shuffle,
        pin_memory=False
    )

    return dataloader
