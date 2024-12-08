
"""
mean + std calculation taken from:
https://kozodoi.me/blog/20210308/compute-image-stats

"""


import imageio
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import torch
from torch.utils import data

from augmentations import *

class cystolith_loader(data.Dataset):

    def __init__(
        self,
        split = 'train',
        img_list_file = '',
        img_size = (512, 512),
        do_resize = (512, 512),
        n_channels = 3,
        n_classes = 2,
        augmentation_params = None,
        mean_rgb = [0.385, 0.411, 0.430],
        img_norm = False,
        debug_info = None,
    ):

        self.split = split
        self.img_list_file = img_list_file
        self.img_size = img_size
        self.do_resize = do_resize
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.augmentations = None
        if augmentation_params is not None:
            self.augmentations_params = augmentation_params
            self.augmentations = BasicAugmentations(self.augmentations_params)

        self.img_norm = img_norm
        self.mean_rgb = mean_rgb

        num_lines = sum(1 for _ in open(self.img_list_file))
        self.image_files = [None] * num_lines
        self.labels = [None] * num_lines
        with open(self.img_list_file, 'r') as f:
            for i, line in enumerate(f):
                values = line.split()
                self.labels[i] = int(values[0])
                self.image_files[i] = values[1]
                if len(values) == 3:    # just for ET's filenames with a whitespace...
                    self.image_files[i] = self.image_files[i] + ' ' + values[2]

        self.save_training_images = 0
        self.save_training_folder = None
        if debug_info is not None:
            self.save_training_images = debug_info.get("save_training_images")
            self.save_training_folder = debug_info.get("save_training_dir")
            if self.save_training_images == 1 and not os.path.exists(self.save_training_folder):
                os.mkdir(self.save_training_folder)

        print("Found %d %s images" % (len(self.image_files), self.split))


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, index):
        lbl = self.labels[index]
        img_file = self.image_files[index]
        img = imageio.v2.imread(img_file)

        if self.save_training_images == 1:
            # file_name = img_file.split(os.sep)[-1]
            file_name = img_file.split('/')[-1]
            file_name = file_name.replace(' ', '')
            file_name = file_name.split('.')[0]
            file_name = self.save_training_folder + os.sep + file_name + '_0.png'
            img_to_save = img.astype(np.uint8)
            imageio.v2.imwrite(file_name, img_to_save)

        if self.do_resize[0] > 0:
            img = Image.fromarray(img).resize((self.do_resize[1], self.do_resize[0]), resample=Image.BICUBIC)
            img = np.array(img)

        img = np.array(img, dtype=np.float)

        if self.save_training_images == 1:
            # file_name = img_file.split(os.sep)[-1]
            file_name = img_file.split('/')[-1]
            file_name = file_name.replace(' ', '')
            file_name = file_name.split('.')[0]
            file_name = self.save_training_folder + os.sep + file_name + '_1.png'
            img_to_save = img.astype(np.uint8)
            imageio.v2.imwrite(file_name, img_to_save)

        img = img.astype(np.float) / 255.0
        if self.img_norm is True:
            img -= self.mean_rgb

        img = img.transpose(2, 0, 1)            # NHWC -> NCHW
        img = torch.from_numpy(img).float()

        if self.augmentations is not None:
            img = self.augmentations(img)

        if self.save_training_images == 1:
            # file_name = img_file.split(os.sep)[-1]
            file_name = img_file.split('/')[-1]
            file_name = file_name.replace(' ', '')
            file_name = file_name.split('.')[0]
            file_name = self.save_training_folder + os.sep + file_name + '_2.png'
            img_to_save = img.numpy()
            img_to_save = img_to_save.transpose(1, 2, 0)
            if self.img_norm is True:
                img_to_save += self.mean_rgb
            img_to_save = 255.0 * img_to_save
            img_to_save = img_to_save.astype(np.uint8)
            imageio.v2.imwrite(file_name, img_to_save)

        return img, lbl, img_file

    def get_img(self, index):
        img_file = self.image_files[index]
        img = imageio.v2.imread(img_file)
        img = img.astype(np.float) / 255.0
        return img

    def get_file_name(self, index):
        img_file = self.image_files[index]
        return img_file