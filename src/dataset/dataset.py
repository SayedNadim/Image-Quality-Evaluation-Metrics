import os
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.utils import natural_keys, allowed_image_extensions, image_reader, find_samples_in_subfolders, \
    default_flist_reader

from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, config, fid=False, fid_gt_or_image='none'):
        super(Dataset, self).__init__()
        self.config = config
        self.fid = fid
        self.fid_gt_or_img = fid_gt_or_image
        if config.dataset_format == 'image':
            if config.dataset_with_subfolders:
                self.gt_samples = find_samples_in_subfolders(config.gt_image_path)
                self.img_samples = find_samples_in_subfolders(config.generated_image_path)
            else:
                self.gt_samples = [os.path.join(config.gt_image_path, x) for x in os.listdir(config.gt_image_path) if
                                   allowed_image_extensions(x)]
                self.img_samples = [os.path.join(config.generated_image_path, x) for x in
                                    os.listdir(config.generated_image_path) if allowed_image_extensions(x)]
        elif config.dataset_format == 'file_list':
            self.gt_samples = default_flist_reader(config.gt_image_path)
            self.img_samples = default_flist_reader(config.generated_image_path)

        self.gt_samples.sort(key=natural_keys)
        self.img_samples.sort(key=natural_keys)
        self.image_shape = config.image_shape[:2]
        self.dataset_name = config.dataset_name
        self.return_dataset_name = config.return_dataset_name

    def __getitem__(self, index):
        img = image_reader(self.img_samples[index])
        gt = image_reader(self.gt_samples[index])

        img = Image.fromarray(img)
        gt = Image.fromarray(gt)

        img = transforms.Resize(self.image_shape)(img)
        gt = transforms.Resize(self.image_shape)(gt)

        img = transforms.ToTensor()(img)
        gt = transforms.ToTensor()(gt)

        if self.fid:
            if self.fid_gt_or_img == 'img':
                return {"images": img}
            elif self.fid_gt_or_img == 'gt':
                return {"images": img}
            else:
                raise KeyError(
                    "FID/IS is true but return type is none. Please make two dataloaders and select img/gt as "
                    "inputs for the dataloaders")
        else:
            if self.return_dataset_name:
                return {"images": img, "gt": gt, "name": self.dataset_name}
            else:
                return {"images": img, "gt": gt}

    def __len__(self):
        return len(self.gt_samples)


def build_dataloader(config, fid, fid_gt_or_image, batch_size,
                     num_workers, shuffle=False):
    dataset = Dataset(
        config=config,
        fid=fid,
        fid_gt_or_image=fid_gt_or_image
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
