from os.path import join

from os import listdir
import os

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir)

def get_val_set(root_dir):
    test_dir = join(root_dir, "validation")

    return DatasetFromFolder(test_dir)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.background_path = join(image_dir, "background")
        self.foreground_path = join(image_dir, "foreground")
        self.label_path = join(image_dir, "original")
        self.image_filenames = [x for x in listdir(self.background_path) if (is_image_file(x) and os.path.isfile(join(self.foreground_path, x)) and os.path.isfile(join(self.label_path, x)))]
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        bg_input = load_img(join(self.background_path, self.image_filenames[index]))
        fg_input = load_img(join(self.foreground_path, self.image_filenames[index]))
        bg_input = self.transform(bg_input)
        fg_input = self.transform(fg_input)
        target = load_img(join(self.label_path, self.image_filenames[index]))
        target = self.transform(target)
        coords = json.load(open(join(self.foreground_path, self.image_filenames[index][:-4] + '_bounding_box.json')))
        coord_data = np.array(coords['center'] + coords['size']).astype(float)

        input = np.concatenate([bg_input, fg_input], axis=0)

        return input, target, coord_data

    def __len__(self):
        return len(self.image_filenames)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGBA')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))