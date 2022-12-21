from __future__ import division
import os
import numpy as np
from imageio import imread
import torch
import torch.utils.data as data
from glob import glob
import json

from datasets import pms_transforms
np.random.seed(0)


class UPS_Custom_Dataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = args.bm_dir
        self.split  = split
        self.args   = args
        self.objs   = ['view_01']
        args.log.printWrite('[%s Data] \t%d views. Root: %s' % (split, len(self.objs), self.root))

    def __getitem__(self, index):
        obj   = self.objs[index]
        img_list = sorted(glob(os.path.join(self.root, f'rgb/*.png')))
        names = [os.path.basename(i) for i in img_list]
        assert [int(ni.split('.')[0]) for ni in names]==[ni+1 for ni in range(len(names))]

        dirs = np.zeros((len(names), 3))
        dirs[:,2] = 1
        ints = np.ones((len(names), 3))

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)
        h, w, c = img.shape
        
        normal = np.zeros((h, w, 3))

        mask = np.array(imread(os.path.join(self.root, f'mask_obj.png')))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask[...,None]/255.
        img  = img * mask.repeat(img.shape[2], 2)

        mi, mj,_ = np.where(mask)
        pad = 15
        crop = (max(0,min(mi)-pad), max(0,min(mj)-pad), min(h,max(mi)+pad), min(w,max(mj)+pad), )
        indices = np.meshgrid(np.arange(crop[0], crop[2]),
                            np.arange(crop[1], crop[3]),
                            indexing='ij')
        img = img[tuple(indices)]
        mask = mask[tuple(indices)]
        normal = normal[tuple(indices)]

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4 
        for k in item.keys():
            item[k] = pms_transforms.imgSizeToFactorOfK(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        item['view'] = obj
        item['crop'] = crop
        item['imres'] = (h,w)
        return item

    def __len__(self):
        return len(self.objs)
