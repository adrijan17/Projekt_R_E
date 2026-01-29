import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch

class WireDataset(Dataset):
    def __init__(self, root_path, test=False, augment=False):
        self.root_path = root_path
        self.augment = augment

        if test:
            self.images = sorted([root_path + "/test_data/" + i
                                  for i in os.listdir(root_path + "/test_data/")])
            self.masks = sorted([root_path + "/test_masks/" + i
                                 for i in os.listdir(root_path + "/test_masks/")])
        else:
            self.images = sorted([root_path + "/train_data/" + i
                                  for i in os.listdir(root_path + "/train_data/")])
            self.masks = sorted([root_path + "/train_masks/" + i
                                 for i in os.listdir(root_path + "/train_masks/")])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        # Promjena na 512x512
        img = TF.resize(img, (512, 512))
        mask = TF.resize(mask, (512, 512))

        # Manipulacija podatcima (samo za treniranje)
        if self.augment:
            # Horizontalno okretanje
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Svjetlina / kontrast (primjena samo na sliku)
            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)

        # Pretvaranje u tenzore
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        # Pretpostavka da je binarna maska
        mask = (mask > 0.5).float()

        return img, mask

    def __len__(self):
        return len(self.images)
