import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        max_intensity = sample['max_intensity']
        name = sample['name']

        return {'image': torch.from_numpy(image), 'image_name': name, 'max_intensity': torch.tensor(max_intensity, dtype=torch.float32)}


class TSDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = [img for img in os.listdir(image_dir) if img.endswith('.tif')]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        img_array = np.array(image)
        img_normalized = img_array / img_array.max()

        # img_normalized_pil = Image.fromarray((img_normalized * 255).astype(np.uint8))
        sample = {'image': img_normalized, 'max_intensity': img_array.max(), 'name': img_path}
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_dataloaders(batch_size=4):
    # 为每个数据集定义不同的归一化转换
    transformA = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.6159], std=[0.2581])  # 假设这是DatasetA的均值和标准差
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transformB = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5949], std=[0.2079])  # 假设这是DatasetB的均值和标准差
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transformAB = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5104], std=[0.2670])  # 假设这是DatasetAB的均值和标准差
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform = transforms.Compose([ToTensor()])

    # 使用不同的归一化参数创建数据集
    datasetA = TSDataset('/root/autodl-tmp/data/2-structures/Training/MTs', transform=transform)
    datasetB = TSDataset('/root/autodl-tmp/data/2-structures/Training/Mito', transform=transform)
    datasetAB = TSDataset('/root/autodl-tmp/data/2-structures/Training/Input', transform=transform)

    dataloaderA = DataLoader(datasetA, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaderB = DataLoader(datasetB, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaderAB = DataLoader(datasetAB, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloaderA, dataloaderB, dataloaderAB

