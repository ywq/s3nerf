import os
import logging
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import imageio
import cv2
import json

logger = logging.getLogger(__name__)

def normalize(x):
    norm = np.linalg.norm(x, axis=-1)
    x = x/(norm[...,None] + 1e-8)
    x[norm==0] = 0
    return x

def get_dataloader(cfg, mode='train', shuffle=True):
    
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']
    kwargs = {}
    split = mode

    dataset = Shapes3dDataset(
        split= split,
        cfg=cfg,
        **kwargs,
    )

    ## dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, 
        shuffle=shuffle, collate_fn=collate_remove_none,
    )

    return dataloader


class Shapes3dDataset(data.Dataset):
    def __init__(self, split='train',
                 cfg=None, **kwargs,
                 ):
        # Attributes
        self.basedir = cfg['dataloading']['data_dir']
        self.obj_name = cfg['dataloading']['obj_name']
        img_dir = os.path.join(self.basedir,self.obj_name)
        self.scale = cfg['dataloading'].get('scale',None)
        self.sres = cfg['dataloading'].get('img_size',None)
        self.capture = cfg['dataloading'].get('capture',None)

        self.paradir = os.path.join(img_dir, 'params.json')
        para = json.load(open(self.paradir))
        imh, imw = para['imhw']
        KK = np.array(para['K']).astype(np.float32)
        if self.capture:
            self.light_pred = np.load(os.path.join(img_dir, 'light_direction_sdps.npy')).astype(np.float32)[0]
            assert len(self.light_pred.shape) == 2
            n_lights = self.light_pred.shape[0]
            assert n_lights == para['n_light']
            dcam = cfg['dataloading'].get('dcam',KK[0,0]*4/imh)
            dlight = cfg['dataloading'].get('dlight', dcam)
            poses = np.array([
                [0,0,1,dcam],
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
            ], dtype=np.float32)
            self.light_pred = np.einsum('ij,kj->ki',poses[:3,:3],self.light_pred).astype(np.float32)
            self.light_direction = self.light_pred * dlight
        else:
            poses = np.array(para['pose_c2w']).astype(np.float32)
            self.light_direction = np.array(para['light_pos']).astype(np.float32) 
            n_lights = self.light_direction.shape[0]

        assert imh == imw
        if self.sres is not None:
            self.scale = imh / self.sres
        elif self.scale is not None:
            self.sres = int(imh / self.scale)
        if self.scale is not None:
            KK[:2,:3] /= self.scale
            imh, imw = self.sres, self.sres

        self.pose0 = poses.copy()
        self.poses = poses.copy()   
        self.poses[:3,1:3]*=-1.
        self.poses = self.poses.astype(np.float32)
        
        imgs, vis = [],[]
        for li in range(n_lights):
            im = np.array(imageio.imread(os.path.join(img_dir,'rgb/{:03d}.png'.format(li+1)),'PNG-FI'))[...,:3]
            if im.dtype == np.uint16:
                im = im/65535.
            elif im.dtype == np.uint8:
                im = im/255.
            if self.scale is not None:
                im = cv2.resize(im.astype(np.float32), (imw, imh), interpolation=cv2.INTER_AREA)
            imgs.append(im.astype(np.float32))
            if not self.capture:
                vis_i = np.array(imageio.imread(os.path.join(img_dir,'visibility/{:03d}.png'.format(li+1)))).astype(np.float32)
                if len(vis_i.shape)!=2: vis_i=vis_i[...,0]
                if self.scale is not None:
                    vis_i = cv2.resize(vis_i, (imw, imh), interpolation=cv2.INTER_NEAREST)
                vis.append(vis_i)

        imgs = np.array(imgs) 
        masks = np.ones_like(imgs[0,...,0]).astype(np.float32) 
        vis = (np.array(vis) / 255.).astype(np.float32) if not self.capture else np.ones_like(imgs[...,0])
        self.mask_obj = None
        
        if self.capture:
            self.normal = np.ones_like(imgs[0])
        else:
            norm0 = np.load(os.path.join(img_dir,'normal.npy'))[...,:3]
            normal = np.einsum('ij,hwi->hwj',self.pose0.astype(np.float64)[:3,:3],norm0.astype(np.float64))
            self.normal = normal.astype(np.float32)
            if self.scale is not None:
                self.normal = normalize(cv2.resize(self.normal, (imw, imh), interpolation=cv2.INTER_AREA)).astype(np.float32)
                
        self.mask_obj = np.array(imageio.imread(os.path.join(img_dir,'mask_obj.png'))).astype(bool).astype(np.float32)
        if self.scale is not None:
            self.mask_obj = cv2.resize(self.mask_obj, (imw, imh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    
        self.imgs = imgs.transpose(0,3,1,2)
        self.normal = self.normal.transpose(2,0,1)
        self.masks = masks
        self.vis = vis

        self.view_idx = np.arange(n_lights)
        self.KK = KK.astype(np.float32)
        print(f'{split}: {self.view_idx.shape[0]} lights')

        
    def __len__(self):
        return len(self.view_idx)

    def __getitem__(self, idx):
        data = {}
        img_idx = self.view_idx[idx]
        data['img']=self.imgs[img_idx]
        data['img.idx']=img_idx
        data['img.world_mat'] = self.poses
        data['img.camera_mat'] = self.KK
        data['img.scale_mat'] = np.eye(4,dtype=np.float32)
        data['img.mask'] = self.masks
        data['img.light'] = self.light_direction[img_idx]
        data['img.vis'] = self.vis[img_idx]
        data['img.normal'] = self.normal
        if self.mask_obj is not None:
            data['img.mask_obj'] = self.mask_obj
        return data


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

