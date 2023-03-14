import torch
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader
from torchvision import transforms
from PIL import Image

class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.img_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        if state == 'test':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        if state == 'val':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()
            self.trans = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.img_dir) / f'{img_id}.jpg'
        img = Image.open(full_path)
        img = self.trans(img)
        return img, label