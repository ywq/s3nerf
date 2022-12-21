import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import dataloading as dl
import model as mdl
import json, imageio

from model.common import arange_pixels
from utils.tools import pose_spherical
from utils.eval_utils import load_light,  vis_light, gen_light_xyz

np.random.seed(42)
torch.manual_seed(42)
from utils.tools import set_debugger
set_debugger()

to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).round().astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()
to_hw = lambda x, h, w: x.reshape(w,h,-1).permute(1,0,2)
rescale = lambda x : (x-x.min())/(x.max()-x.min())

# Arguments
parser = argparse.ArgumentParser(
    description='Testing of S^3-NeRF'
)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--gpu', type=int, help='gpu')
parser.add_argument('--obj_name', type=str, default='bunny',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--exp_folder', type=str, default='out',)
parser.add_argument('--test_out_dir', type=str, default='test_out',)
parser.add_argument('--load_iter', type=int, default=None)
parser.add_argument('--type', type=str, default='light',)
parser.add_argument('--chunk', type=int, default=1024)
parser.add_argument('--envmap_path', type=str, default='envmap')
parser.add_argument('--envmap_id', default=3, type=int,)
parser.add_argument('--edit_albedo', default=False, action="store_true", help='If set, edit albedo')
parser.add_argument('--edit_specular', default=False, action="store_true", help='If set, edit specular')
parser.add_argument('--basis', default=None, type=int, help='specular basis')
parser.add_argument('--color', default=None, type=str, help='albedo color')
parser.add_argument('--save_npy', action='store_true', default=False)
args = parser.parse_args()

cfg = dl.load_config(os.path.join(args.exp_folder,args.obj_name, args.expname, 'config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

test_out_path = os.path.join(args.test_out_dir, args.obj_name, args.expname)
nsub = args.type
vsub_img, vsub_npy = '',''

if 'env' in args.type:
    light_h = 16  # 16 used in nerfactor
    args.envmap_path = os.path.join(args.envmap_path,'indoor-{0:02d}/indoor-{0:02d}.exr'.format(args.envmap_id))
    env_light = load_light(args.envmap_path, light_h=light_h)
    ldis = 100
    env_light *= 1./env_light.sum() * ldis**2 *10
    envmap_name = os.path.basename(args.envmap_path)[:-len('.hdr')]
    nsub = f'envmap/{envmap_name}'
    os.makedirs(os.path.join(test_out_path,nsub), exist_ok=True)
    os.system("""cp -r {0} "{1}" """.format(args.envmap_path, os.path.join(test_out_path,nsub)))
    _ = vis_light(env_light, os.path.join(os.path.join(test_out_path,nsub),envmap_name+'.png'), h=light_h*8)
    lxyz, lareas = gen_light_xyz(light_h, 2*light_h, envmap_radius=ldis)
    lxyz = lxyz.reshape(-1,3).astype(np.float32)
    env_light = env_light.reshape(-1,3).astype(np.float32)
elif args.edit_albedo or args.edit_specular:
    args.type = 'edit'
    albedo_new,basis_new=None,None
    nexp = ''
    if args.edit_albedo:
        if args.color is None:
            albedo_new = np.random.choice(range(128),size=3)
            nexp += '#{:02x}{:02x}{:02x}'.format(*list(albedo_new))
        else:
            albedo_new = np.array([int(args.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])
            albedo_new = albedo_new.astype(np.float32)/5.
            nexp = args.color
        albedo_new = (albedo_new/255.).astype(np.float32)
    if args.edit_specular:
        if args.basis is None:
            basis_new = np.random.choice(range(9))
        else:
            basis_new = args.basis
        nexp = f'sg{basis_new+1}' if nexp=='' else nexp+f'_sg{basis_new+1}'
    nsub = f'edit/{nexp}'
    for savedir in ['rgb', 'visibility', 'specular']:
        os.makedirs(os.path.join(test_out_path, nsub, savedir, 'img'), exist_ok=True)
        if args.save_npy:
            os.makedirs(os.path.join(test_out_path, nsub, savedir, 'npy'), exist_ok=True)

else:
    for savedir in ['rgb', 'visibility', 'specular']:
        os.makedirs(os.path.join(test_out_path, nsub, savedir, 'img'), exist_ok=True)
        if args.save_npy:
            os.makedirs(os.path.join(test_out_path, nsub, savedir, 'npy'), exist_ok=True)
    if 'view' in args.type:
        for savedir in ['mask', 'depth', 'normal', 'albedo']:
            os.makedirs(os.path.join(test_out_path, nsub, savedir, 'img'), exist_ok=True)
            if args.save_npy or savedir in ['depth','normal']:
                os.makedirs(os.path.join(test_out_path, nsub, savedir, 'npy'), exist_ok=True)

# init network
capture = cfg['dataloading'].get('capture',False)
model = mdl.NeuralNetwork(cfg)
if 'view' in args.type and not capture:
    model.light_int_src *= 2

# init renderer
renderer = mdl.Renderer(model, cfg, device=device)
renderer.render_fine = True
renderer.render_fine_iter = -1

# init checkpoints and load
out_dir = os.path.join(args.exp_folder, args.obj_name, args.expname)
checkpoint_io = mdl.CheckpointIO(os.path.join(out_dir,'models'), model=model)

try:
    load_dict = checkpoint_io.load(f'model_{args.load_iter}.pt' if args.load_iter else 'model.pt')
except FileExistsError:
    load_dict = dict()
it = load_dict.get('it', 100000)

f = os.path.join(test_out_path, 'config.yaml')
with open(f, 'w') as file:
    file.write(open(os.path.join(out_dir, 'config.yaml'), 'r').read())

basedir = cfg['dataloading']['data_dir']
img_dir = os.path.join(basedir, args.obj_name)
paradir = os.path.join(img_dir, 'params.json')
para = json.load(open(paradir))
KK = np.array(para['K']).astype(np.float32)
h,w = para['imhw']
scale = cfg['dataloading'].get('scale',None)
sres = cfg['dataloading'].get('img_size',None)
if sres is not None:
    scale = h / sres
elif scale is not None:
    sres = int(h / scale)
if scale is not None:
    KK[:2,:3] /= scale
    h, w = sres, sres
renderer.im_res = (h,w)
if capture:
    dcam = cfg['dataloading'].get('dcam',KK[0,0]*4/h)
    dlight = cfg['dataloading'].get('dlight', dcam)
    poses = np.array([
                [0,0,1,dcam],
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
            ], dtype=np.float32)
    light_pos = np.random.normal(size=(1000,3))
    light_pos = light_pos / np.linalg.norm(light_pos,axis=-1,keepdims=True)
    light_pos = light_pos[light_pos[:,-1]>-0.2]
    light_pos = light_pos[(light_pos*poses[:3,2]).sum(-1)>0.5].astype(np.float32)
    if len(light_pos)>8: light_pos = light_pos[:8]
    light_pos = light_pos * dlight
else:
    poses = np.array(para['pose_c2w']).astype(np.float32)
    light_pos = np.array(para['light_pos_test']).astype(np.float32) 
    if args.type in ['view']:
        assert 'pose_c2w_test' in para
        poses_test = np.array(para['pose_c2w_test']).astype(np.float32) 
        poses_test[:3,1:3]*=-1.

pose0 = poses.copy()
poses[:3,1:3]*=-1.
pose_ori = torch.tensor(poses.copy()).to(device)
    
if 'env' in args.type:
    light_pos = lxyz.astype(np.float32)
    vis_pre = None
    if os.path.exists(os.path.join(test_out_path,'envmap/vis_all.npy')):
        vis_pre = np.load(os.path.join(test_out_path,'envmap/vis_all.npy')).astype(np.float32)
        vis_pre = torch.tensor(vis_pre).to(device)
elif args.type=='edit':
    edit_mask = np.array(imageio.imread(os.path.join(img_dir,'mask_obj.png'))).astype(bool)
    if len(edit_mask.shape)>2:
        mask_obj = edit_mask[...,0]
    edit_mask = torch.tensor(edit_mask).to(device)
elif 'light' in args.type or 'view' in args.type:
    if args.type in ['view']:
        light_pos = pose_ori[:3,3].cpu().numpy()
        if capture:
            tpara = json.load(open(os.path.join(out_dir,'test.json'), 'r'))
            poses = np.array(tpara['view']).astype(np.float32)
            poses[:,:3,1:3]*=-1.
        else:
            poses = poses_test
    elif 'render_view' in args.type:
        light_pos = pose_ori.cpu().numpy()[:3,3]
        agl = np.tanh(np.abs(pose_ori.cpu().numpy()[2,2])/np.linalg.norm(pose_ori.cpu().numpy()[:2,2],axis=-1))/np.pi*180
        poses = torch.stack([pose_spherical(angle, -agl, 4) for angle in np.linspace(180,0,20)], 0)
        poses[:,:3,1:3]*=-1.
else:
    raise ValueError

light_pos = torch.tensor(light_pos).to(device)
poses = torch.tensor(poses).to(device)

datas = poses if 'view' in args.type else light_pos
camera_mat = torch.tensor(KK).to(device)[None,]
scale_mat = torch.eye(4,dtype=torch.float32).to(device)

if 'env' in args.type:
    world_mat = poses[None]
    light_dim = light_h**2*2

    p_loc, pixels = arange_pixels(resolution=(h, w))
    pixels = pixels.to(device)
    p_loc = p_loc.to(device)
    light_src = light_pos
    renderer.light_int_src = torch.tensor(env_light).to(device)
    model.env = True
    renderer.env = True
    with torch.no_grad():
        rgb_pred, vis_pred, vis_all = [],[],[]
        for ii, pixels_i in enumerate(tqdm(torch.split(p_loc, args.chunk, dim=1),ncols=120,desc="Envmap",bar_format='{desc}: {percentage:3.0f}%|{bar}|')):
            vis_pre_i = torch.split(vis_pre.permute(0,2,1).reshape(light_dim,-1), args.chunk, dim=1)[ii] if vis_pre is not None else None
            out_dict = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                        add_noise=False, eval_=True, it=it, light_src=light_src, 
                        novel_view='view' in args.type, view_ori=pose_ori[None,], vis_pre=vis_pre_i)
            rgb_pred.append(out_dict.get('rgb_fine',None))
            vis_pred.append(out_dict['vis'])
            vis_all.append(out_dict.get('vis_all',None))
        rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h, w)).astype(np.float32)
        vis_pred = to_numpy(to_hw(torch.cat(vis_pred, dim=1),h, w))[...,0].astype(np.float32)
        if vis_all[0] is not None:
            vis_all = to_numpy(torch.cat(vis_all, dim=1).reshape(light_dim,w,h).permute(0,2,1)).astype(np.float32)
            np.save(os.path.join(test_out_path,'envmap/vis_all.npy'), vis_all)
    img = Image.fromarray(to_img(rgb_pred))
    img.save(os.path.join(test_out_path,nsub,'rgb.png'))
    img = Image.fromarray(to_img(vis_pred))
    img.save(os.path.join(test_out_path,nsub,'visibility.png'))
    if args.save_npy:
        np.save(os.path.join(test_out_path,nsub,'rgb.npy'), rgb_pred)
        np.save(os.path.join(test_out_path,nsub,'visibility.npy'), vis_pred)

else:
    for di, data in enumerate(tqdm(datas, ncols=120)):
        lidx_ori = di
        p_loc, pixels = arange_pixels(resolution=(h, w))
        pixels = pixels.to(device)
        p_loc = p_loc.to(device)
        light_src = light_pos[None,] if 'view' in args.type else light_pos[di:di+1]
        world_mat = poses[di:di+1] if 'view' in args.type else poses[None,]
        if 'view' in args.type:
            vsub_npy = f'/npy/{lidx_ori+1:03d}'
            vsub_img = f'/img/{lidx_ori+1:03d}'

        with torch.no_grad():
            rgb_pred, albedo_pred, spec_pred, norm_pred, mask_pred, depth_pred, vis_pred = [],[],[],[],[],[],[]
            for ii, pixels_i in enumerate(torch.split(p_loc, args.chunk, dim=1)):
                if args.type=='edit':
                    renderer.edit_mask = torch.split(edit_mask.permute(1,0).reshape(-1), args.chunk, dim=0)[ii]
                    if albedo_new is not None:
                        model.albedo_new = torch.tensor(albedo_new).to(device)
                    if basis_new is not None:
                        model.basis_new = basis_new
                out_dict = renderer(pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                            add_noise=False, eval_=True, it=it, light_src=light_src, 
                            novel_view='view' in args.type, view_ori=pose_ori[None,])
                rgb_pred.append(out_dict.get('rgb_fine',None))
                albedo_pred.append(out_dict.get('albedo_fine',None))
                spec_pred.append(out_dict.get('specular_fine',None))
                norm_pred.append(out_dict.get('normal_pred',None))
                mask_pred.append(out_dict.get('mask_pred',None))
                depth_pred.append(out_dict['depth'])
                vis_pred.append(out_dict['vis'])
            rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h, w)).astype(np.float32)
            albedo_pred = to_numpy(to_hw(torch.cat(albedo_pred, dim=1),h, w)).astype(np.float32)
            spec_pred = to_numpy(to_hw(torch.cat(spec_pred, dim=1),h, w)).astype(np.float32)
            norm_pred = to_numpy(to_hw(torch.cat(norm_pred, dim=1),h, w)).astype(np.float32)
            depth_pred = to_numpy(to_hw(torch.cat(depth_pred, dim=1),h, w))[...,0].astype(np.float32)
            vis_pred = to_numpy(to_hw(torch.cat(vis_pred, dim=1),h, w))[...,0].astype(np.float32)
            mask_pred = to_numpy(to_hw(torch.cat(mask_pred, dim=0),h, w))[...,0]

            img = Image.fromarray(to_img(rgb_pred))
            img.save(os.path.join(test_out_path,nsub,'rgb/img/{:03d}.png'.format(lidx_ori+1)))
            img = Image.fromarray(to_img(vis_pred))
            img.save(os.path.join(test_out_path,nsub,'visibility/img/{:03d}.png'.format(lidx_ori+1)))
            img = Image.fromarray(to_img(spec_pred))
            img.save(os.path.join(test_out_path,nsub,'specular/img/{:03d}.png'.format(lidx_ori+1)))
            if args.save_npy:
                np.save(os.path.join(test_out_path,nsub,'rgb/npy/{:03d}.npy'.format(lidx_ori+1)), rgb_pred)
                np.save(os.path.join(test_out_path,nsub,'visibility/npy/{:03d}.npy'.format(lidx_ori+1)), vis_pred)
                np.save(os.path.join(test_out_path,nsub,'specular/npy/{:03d}.npy'.format(lidx_ori+1)), spec_pred)
            
            if ('light' in args.type or args.type in ['edit']) and di > 0:
                continue

            img = Image.fromarray(to_img(mask_pred))
            img.save(os.path.join(test_out_path,nsub,f'mask{vsub_img}.png'))
            depth_pred_scale = np.ones((h, w)).astype(np.float32)
            depth_pred_scale[mask_pred] = rescale(depth_pred[mask_pred])
            img = Image.fromarray(to_img(depth_pred_scale))
            img.save(os.path.join(test_out_path,nsub,f'depth{vsub_img}.png'))
            img = Image.fromarray(to_img(albedo_pred))
            img.save(os.path.join(test_out_path,nsub,f'albedo{vsub_img}.png'))
            norm_trans = np.einsum('ij,hwi->hwj',to_numpy(world_mat)[0,:3,:3].astype(np.float64)*np.array([1,-1,-1]),norm_pred.astype(np.float64))
            img = Image.fromarray(to_img(norm_trans/2.+0.5))
            img.save(os.path.join(test_out_path,nsub,f'normal{vsub_img}.png'))
            np.save(os.path.join(test_out_path,nsub,f'normal{vsub_npy}.npy'), norm_pred)
            np.save(os.path.join(test_out_path,nsub,f'depth{vsub_npy}.npy'), depth_pred)
            if args.save_npy:
                np.save(os.path.join(test_out_path,nsub,f'mask{vsub_npy}.npy'), mask_pred)
                np.save(os.path.join(test_out_path,nsub,f'albedo{vsub_npy}.npy'), albedo_pred)
