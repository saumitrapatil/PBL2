import torch
from torch.utils.data import Dataset
import os
import cv2

class LowLightDataset(Dataset):
    def __init__(self, low_light_dir, normal_dir, transform=None):
        self.low_light_files = sorted(os.listdir(low_light_dir))
        self.normal_files = sorted(os.listdir(normal_dir))
        self.low_light_dir = low_light_dir
        self.normal_dir = normal_dir
        self.transform = transform

    def __len__(self):
        return len(self.low_light_files)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_light_dir, self.low_light_files[idx])
        normal_path = os.path.join(self.normal_dir, self.normal_files[idx])

        low_img = cv2.imread(low_path)
        normal_img = cv2.imread(normal_path)

        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB) / 255.0
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB) / 255.0

        low_img = torch.tensor(low_img.transpose(2, 0, 1), dtype=torch.float32)
        normal_img = torch.tensor(normal_img.transpose(2, 0, 1), dtype=torch.float32)

        return low_img, normal_img