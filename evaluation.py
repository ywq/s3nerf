import os, sys
import argparse
import numpy as np
import imageio, json
from scipy import ndimage
from dataloading import load_config

import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')

from utils.metrics import MAE, PSNR, zerr
  

# Arguments
parser = argparse.ArgumentParser(
    description='Evaluation'
)
parser.add_argument('--obj_name', type=str, default='bunny',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--type', type=str, default='light',)
parser.add_argument('--obj_only', action='store_true', default=False,)
parser.add_argument('--depth_align', action='store_true', default=False,)
parser.add_argument('--test_out_dir', type=str, default='test_out',)
args = parser.parse_args()

test_out_path = os.path.join(args.test_out_dir, args.obj_name, args.expname)
cfg = load_config(os.path.join(test_out_path, 'config.yaml'))
basedir = cfg['dataloading']['data_dir']
img_dir = os.path.join(basedir, args.obj_name)
para = json.load(open(os.path.join(img_dir, 'params.json')))
poses = np.array(para['pose_c2w']).astype(np.float32)
KK = np.array(para['K']).astype(np.float32)
imh,imw = para['imhw']
plog = f'[{args.obj_name}][{args.expname}]:  '

mask_obj = np.array(imageio.imread(os.path.join(img_dir,'mask_obj.png'))).astype(bool)
mask_obj = ndimage.binary_erosion(mask_obj, structure=np.ones((3,3))).astype(bool)
normal_gt = np.load(os.path.join(img_dir,'normal.npy')).astype(np.float32)
depth_gt = np.load(os.path.join(img_dir,'depth.npy')).astype(np.float32)

normal = np.load(os.path.join(test_out_path,args.type,'normal.npy')).astype(np.float32)
depth = np.load(os.path.join(test_out_path, args.type,'depth.npy'))
mask_obj_err = mask_obj if args.obj_only else None
nerr = MAE(normal,normal_gt,mask_obj_err)[0]
derr= zerr(depth,depth_gt,KK,imh,imw,mask_obj=mask_obj_err, align=args.depth_align,predz=False,gtz=True)
plog += f'Normal MAE{" (obj_only)" if args.obj_only else ""}:  {nerr:.2f}   '
plog += f'Depth L1{" (align)" if args.depth_align else ""}{" (obj_only)" if args.obj_only else ""}:  {derr*100:.2f}   '

if 'light' in os.listdir(img_dir):
    img_data = []
    for lidx in range(len(para['light_pos_test'])):
        lgt = imageio.imread(os.path.join(img_dir, 'light',f'rgb/{lidx+1:03d}.png')).astype(np.float32)/255.
        limg = imageio.imread(os.path.join(test_out_path,'light',f'rgb/img/{lidx+1:03d}.png')).astype(np.float32)/255.
        img_data.append(PSNR(limg,lgt))
    plog += f'PSNR:  {np.array(img_data).mean():.2f}  '

print(plog)