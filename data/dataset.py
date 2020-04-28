# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np
import PIL
import torch

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:,::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class DataSet(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True,train_txt=None,test_txt=None,invalid_txt="./imagesTxtPath/invalid.txt",image_w=32,image_h=64,train_mode='default'):
        '''
        Get images, divide into train/val set
        '''   
        self.train = train
        self.images_root = root

        self.train_txt = train_txt
        self.test_txt = test_txt
        self.invalid_txt = invalid_txt
        self.image_w = image_w
        self.image_h = image_h

        self.data_len = self._read_txt_file()
    
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if not train:  
                if train_mode == '32x64':
                    self.transforms = T.Compose([
                        T.Resize((64 + 5, 32 + 5)),
                        T.CenterCrop((image_h,image_w)),
                        T.ToTensor(),
                        normalize
                    ])
                else:
                    self.transforms = T.Compose([
			            T.Resize((32,32)),
                        T.CenterCrop((image_h,image_w)),
                       T.ToTensor(),
                        normalize
                        #ToBGRTensor()
                    ])

            else:
                if train_mode == 'default':
                    self.transforms = T.Compose([
			            T.Resize((32+2,32+2)),#32+3,16+2
		                T.RandomRotation(30),
                        T.RandomCrop((image_h,image_w)),
		        #T.CenterCrop((image_h, image_w)),
                        T.RandomHorizontalFlip(),
                        #T.RandomVerticalFlip(),
                        #T.RandomRotation(180),
                        T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                        T.ToTensor(),
                        normalize
			#ToBGRTensor()
                    ])
                elif train_mode == '32x64':
                    self.transforms = T.Compose([
                        T.Resize((64 + 5, 32 + 5)),
                        T.RandomRotation(30),
                        T.RandomCrop((64,32)),
                        #T.CenterCrop((image_h, image_w)),
                        T.RandomHorizontalFlip(),
                        #T.RandomVerticalFlip(),
                        #T.RandomRotation(180),
                        T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                        T.ToTensor(),
                        normalize
                    ])
                elif train_mode == '64x128':
                    self.transforms = T.Compose([
                        T.Resize((128 + 10, 64 + 10)),
                        T.RandomRotation(30),
                        T.RandomCrop((128,64)),
                        #T.CenterCrop((image_h, image_w)),
                        T.RandomHorizontalFlip(),
                        #T.RandomVerticalFlip(),
                        #T.RandomRotation(180),
                        T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                        #T.ToTensor(),
                        #normalize
                        ToBGRTensor()
                    ])
                elif train_mode == '32x64to64x128':
                    self.transforms = T.Compose([
                        T.Resize((64 + 5, 32 + 5)),
                        T.Resize((128 + 10, 64 + 10)),
                        T.RandomRotation(30),
                        T.RandomCrop((128,64)),
                        #T.CenterCrop((image_h, image_w)),
                        T.RandomHorizontalFlip(),
                        #T.RandomVerticalFlip(),
                        #T.RandomRotation(180),
                        T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                        T.ToTensor(),
                        normalize
                    ])
    def get_data_len(self):
        return self.data_len


                
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = self.train_txt
        else:
            txt_file = self.test_txt

        #invalid_txt_file = "./images/invalid.txt"
        invalid_txt_file = self.invalid_txt
        invalid_path = []
        with open(invalid_txt_file, 'r') as invalid_f:
            for line in invalid_f.readlines():
                invalid_path.append(line.strip())
        invalid_f.close()
        # print(invalid_path)

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                if item[0] not in invalid_path:
                    self.images_path.append(item[0])
                    self.images_labels.append(item[1])

        return len(self.images_path)

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_path[index]
        # print(img_path)
        label = self.images_labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, int(label)
    
    def __len__(self):
        return len(self.images_path)
