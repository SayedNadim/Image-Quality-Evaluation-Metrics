from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for paths
import os
import sys

# for tensor operations
import torch

# for image reading purpose
import cv2
import skimage.io as sio
import imageio
import numpy as np
from PIL import Image
import ntpath
import skimage.feature as feature

# for sorting
import re
from natsort import natsorted
import fnmatch


# helper functions

# file helpers
def path_leaf(path, image=True, cut_off_value=-4):
    head, tail = ntpath.split(path)
    if image:
        return tail[:cut_off_value] or ntpath.basename(head)
    else:
        return tail or ntpath.basename(head)


def default_flist_reader(flist):
    """
    flist format: img_path\nimg_path\n ...
    """
    im_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            dirpath = line.strip().split()
            im_list.append(dirpath)

    return natsorted(im_list)


# image reader
def image_reader(path, is_color=True, lib='cv2'):
    """
    Reads an image from the given path and returns the color or gray image using preferred library.
    :param path: Path of the image file
    :param is_color: Condition for the image is to be read as color image or grayscale image
    :param lib: Preferred image reading library. Available: cv2, skimage, pil and imageion. Note that, color image from
                cv2 does not comply with other libraries.
    :return: image value (numpy array)
    """
    if lib == 'skimage':
        if is_color:
            img = sio.imread(path)
        else:
            img = sio.imread(path, as_gray=True)
            img = img[:, :, np.newaxis]
    # TODO make cv2 image = image from other libraries. Currently all libraries match except cv2.
    #  I think, this is due to my implementation issue, maybe...
    elif lib == 'cv2':
        if is_color:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
    elif lib == 'imageio':
        if is_color:
            img = imageio.imread(path)
        else:
            img = imageio.imread(path)
            gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            img = gray(img)
            img = img[:, :, np.newaxis]
    elif lib == 'pil':
        if is_color:
            img = np.array(Image.open(path))
        else:
            img = np.array(Image.open(path).convert('L'))
            img = img[:, :, np.newaxis]

    else:
        raise KeyError(
            "Invalid library selected. Available libraries are: \'skimage\', \'cv2\',\'PIL\', and \'imageio\' ")

    return img.astype(np.uint8)


def canny_edge_map(image):
    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    image = np.array(image)
    out = np.uint8(feature.canny(image, sigma=1, ) * 255)
    return out


def resize_image(img, height=256, width=256, centerCrop=False):
    imgh, imgw, imgc = img.shape
    if imgh <= height or imgw <= width:
        img = cv2.resize(img, (width * 1.5, height * 1.5, imgc))

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = cv2.resize(img, (height, width, imgc))
    return img


def get_crop_params(image, crop_size, crop_method='random_crop'):
    crop_height, crop_width = crop_size
    start_x, end_x, start_y, end_y = 0, 0, 0, 0
    if image.shape[0] <= crop_height or image.shape[1] <= crop_width:
        image = cv2.resize(image, (crop_width * 2, crop_height * 2))

    if crop_method == 'random_crop':
        max_x = image.shape[1] // 2 - crop_width
        max_y = image.shape[0] // 2 - crop_height

        start_x = np.random.randint(0, max_x)
        end_x = start_x + crop_width
        start_y = np.random.randint(0, max_y)
        end_y = start_y + crop_height
    elif crop_method == 'center_crop':
        start_x = (image.shape[1] // 2 - crop_width // 2)
        end_x = start_x + crop_width
        start_y = (image.shape[0] // 2 - crop_height // 2)
        end_y = start_y + crop_height

    return {'top_left': (start_x, start_y), 'right_bottom': (end_x, end_y)}


def crop_image(image, crop_pos, crop_size=(256, 256), save_image=False):
    if crop_pos == None:
        crop_pos = get_crop_params(image, crop_size)
    top_left = crop_pos.get('top_left')
    right_bottom = crop_pos.get('right_bottom')
    cropped_image = image[top_left[1]:right_bottom[1], top_left[0]:right_bottom[0], ...]
    if save_image:
        imageio.imsave('./image.png', image)
        imageio.imsave('./cropped_image.png', cropped_image)
    return cropped_image


# tensor functions
def normalize_tensor(tensor_image):
    return tensor_image.mul_(2).add_(-1)


def denormalize_tensor(tensor_image):
    return tensor_image.add_(1).div_(2)


def reduce_mean(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.mean(tensor, dim=i, keepdim=keepdim)
    return tensor


def reduce_sum(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.sum(tensor, dim=i, keepdim=keepdim)
    return tensor


def reduce_std(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.std(tensor, dim=i, keepdim=keepdim)
    return tensor


# array image functions
def allowed_image_extensions(filename):
    """
    Returns image files if they have the allowed image extensions
    :param filename: image file
    :return: image file
    """
    img_ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_ext)


def find_samples_in_subfolders(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    samples = []
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if allowed_image_extensions(fname):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    # samples.append(item)
                    samples.append(path)
    return natsorted(samples)


def find_folders(rootdir):
    samples =  [dI for dI in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, dI))]
    return natsorted(samples)

# sorting help
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]
