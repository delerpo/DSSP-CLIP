"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
from torchvision import transforms
import math
import torch

class VisaDataset(Dataset):
    def __init__(self, root, train=True, category=None, fewshot=0, transform=None, gt_target_transform=None, aug_data=True):
        super(VisaDataset, self).__init__()
        self.categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                           'pcb4', 'pipe_fryum']
        self.train = train
        self.category = category
        self.fewshot = fewshot
        self.root = os.path.join(root, 'VisA_20220922')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        # ------数据增强------
        self.aug_data = aug_data
        self.color_transforms = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.7),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.7),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], p=0.7)
        ])
        self.transforms_list = [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
        self.random_transform = transforms.Compose(self.transforms_list)
        # --------------
        self.preprocess()
        self.update(category=category) 
        self.dataset_name = "visa"
       
        
    def preprocess(self):
        self.csv_data = pd.read_csv(os.path.join(self.root, 'split_csv/1cls.csv'), header=0)
        columns = self.csv_data.columns
        images_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        gt_paths = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        labels = {'train': {category : [] for category in self.categories}, 'test': {category : [] for category in self.categories}}
        for category in self.categories:
            cls_data = self.csv_data[self.csv_data[columns[0]] == category]
            for phase in ['train', 'test']:
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                for _, row in cls_data_phase.iterrows():
                    img_path = row[columns[3]]
                    label = 1 if row[columns[2]] == 'anomaly' else 0
                    mask_path = row[columns[4]] if row[columns[2]] == 'anomaly' else None
                    images_paths[phase][category].append(img_path)
                    gt_paths[phase][category].append(mask_path)
                    labels[phase][category].append(label)
        self.img_paths = images_paths
        self.gt_paths = gt_paths
        self.labels = labels
        
    def update(self, category=None):
        self.category = category
        tot_img_paths, tot_gt_paths, tot_img_classes, tot_img_labels = [], [], [], []
        if self.train:
            phase = 'train'
        else:
            phase = 'test'
        if self.category is not None:
            tot_img_paths = self.img_paths[phase][self.category]
            tot_gt_paths = self.gt_paths[phase][self.category]
            tot_img_classes = [self.category] * len(tot_img_paths)
            tot_img_labels = self.labels[phase][self.category]
        else:
            for category in self.categories:
                tot_img_paths.extend(self.img_paths[phase][category])
                tot_gt_paths.extend(self.gt_paths[phase][category])
                tot_img_classes.extend([category] * len(self.img_paths[phase][category]))
                tot_img_labels.extend(self.labels[phase][category])
        
        self.cur_img_paths = tot_img_paths 
        self.cur_gt_paths = tot_gt_paths
        self.cur_img_categories = tot_img_classes
        self.cur_img_labels = tot_img_labels
                
        if self.fewshot != 0:

            randidx = np.random.choice(len(self.cur_img_paths), size=self.fewshot, replace=False)
            self.cur_img_paths = [self.cur_img_paths[idx] for idx in randidx]
            self.cur_gt_paths = [self.cur_gt_paths[idx] for idx in randidx]
            self.cur_img_labels = [self.cur_img_labels[idx] for idx in randidx]
            self.cur_img_categories = [self.cur_img_categories[idx] for idx in randidx]
    
    def __len__(self):
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        category = self.cur_img_categories[idx]
        img_path = os.path.join(self.root, self.cur_img_paths[idx])
        label = self.cur_img_labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.cur_gt_paths[idx] is not None:
            mask_path = os.path.join(self.root, self.cur_gt_paths[idx])
            gt = np.array(Image.open(mask_path).convert('L'))
            gt[gt != 0] = 255
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        if self.transform is not None:
            if self.aug_data:
                img = self.color_transforms(img)
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        if self.aug_data:
            # 拼接: [3, H, W] + [1, H, W] -> [4, H, W]
            transform_tensor = torch.cat([img, gt], dim=0)
            # 应用空间增强
            transform_tensor = self.random_transform(transform_tensor)
            # 拆分
            img = transform_tensor[0:3, :, :]
            gt = transform_tensor[3:4, :, :]
            gt = (gt > 0.5).float()

        return img, label, gt, category, img_path
    
    
    