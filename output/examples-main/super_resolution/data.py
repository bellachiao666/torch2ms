import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return ms.dataset.transforms.Compose([
        ms.dataset.vision.CenterCrop(crop_size),
        ms.dataset.vision.Resize(crop_size // upscale_factor),
        ToTensor(),
    ])  # 'torchvision.transforms.Resize'默认值不一致(position 1): PyTorch=<InterpolationMode.BILINEAR: 'bilinear'>, MindSpore=2;; 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def target_transform(crop_size):
    return ms.dataset.transforms.Compose([
        ms.dataset.vision.CenterCrop(crop_size),
        ToTensor(),
    ])  # 'torchvision.transforms.ToTensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
