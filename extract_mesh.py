import os
import argparse
import time, json
import numpy as np
import torch.nn.functional as F
from model.common import origin_to_world,image_points_to_ray

import torch
from scipy.spatial.transform import Rotation as R

from dataloading import load_config
from model.checkpoints import CheckpointIO
from model.network import NeuralNetwork
from model.extracting import Extractor3D
import _pickle

from utils.tools import set_debugger
set_debugger()
torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--gpu', type=int, help='gpu')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--upsampling-steps', type=int, default=-1,
                    help='Overrites the default upsampling steps in config')
parser.add_argument('--refinement-step', type=int, default=-1,
                    help='Overrites the default refinement steps in config')
parser.add_argument('--obj_name', type=str, default='bunny',)
parser.add_argument('--expname', type=str, default='test_1',)
parser.add_argument('--exp_folder', type=str, default='out',)
parser.add_argument('--test_out_dir', type=str, default='test_out',)
parser.add_argument('--load_iter', type=int, default=None)
parser.add_argument('--mesh_extension', type=str, default='obj')
parser.add_argument('--clip', action='store_true', default=False,)
parser.add_argument('--pad', type=float, default=2.4,)
parser.add_argument('--desk', type=float, default=-1.2,)  # valid regions
parser.add_argument('--wall', type=float, default=-2.2,)
parser.add_argument('--obj_only', action='store_true', default=False,)
parser.add_argument('--obj_scale', type=float, default=0.7,)

args = parser.parse_args()
cfg = load_config(os.path.join(args.exp_folder,args.obj_name, args.expname, 'config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if args.upsampling_steps != -1:
    cfg['extraction']['upsampling_steps'] = args.upsampling_steps
if args.refinement_step != -1:
    cfg['extraction']['refinement_step'] = args.refinement_step

test_out_path = os.path.join(args.test_out_dir, args.obj_name, args.expname)
os.makedirs(test_out_path,exist_ok=True)

# Model
model = NeuralNetwork(cfg)
out_dir = os.path.join(args.exp_folder, args.obj_name, args.expname)
checkpoint_io = CheckpointIO(os.path.join(out_dir,'models'), model=model)
load_dict = checkpoint_io.load(f'model_{args.load_iter}.pt' if args.load_iter else 'model.pt')
it = load_dict.get('it', 100000)

args.pad = abs(args.wall) * 2 - 2
# Generator
generator = Extractor3D(
    model, resolution0=cfg['extraction']['resolution'], 
    upsampling_steps=cfg['extraction']['upsampling_steps'], 
    device=device, padding=args.pad,
)
generator.desk = args.desk
generator.wall = args.wall

basedir = cfg['dataloading']['data_dir']
para = json.load(open(os.path.join(basedir, args.obj_name, 'params.json')))
KK = np.array(para['K']).astype(np.float32)
h,w = para['imhw']
if cfg['dataloading'].get('capture',False):
    dcam = cfg['dataloading'].get('dcam',KK[0,0]*4/h)
    dlight = cfg['dataloading'].get('dlight', dcam)
    poses = np.array([
                [0,0,1,dcam],
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
            ], dtype=np.float32)
else:
    poses = np.array(para['pose_c2w']).astype(np.float32)
poses[:3,1:3]*=-1.

KK = torch.tensor(KK).to(device)[None,]
poses = torch.tensor(poses.copy()).to(device)[None,]
p_corner = torch.tensor([[
    [0,0], [w-1,0], [w-1,h-1], [0,h-1]
]],dtype=torch.float32).to(device)

ray_corner = image_points_to_ray(p_corner, KK, poses).view(-1,3)
ray_corner = F.normalize(ray_corner,dim=-1)
ray_corner_2 = torch.cat([ray_corner[1:],ray_corner[:1]],dim=0)
n_plane = F.normalize(torch.cross(ray_corner,ray_corner_2))
scale_mat = torch.eye(4,dtype=torch.float32).to(device)
generator.camloc = origin_to_world(1, KK, poses, scale_mat).reshape(-1).cpu()
generator.n_plane = n_plane.cpu()

if args.obj_only:
    ddesk = -1
    generator.padding = (args.obj_scale + 0.2) * 2 - 2
    generator.shift = [0, 0, args.obj_scale + ddesk]

# Generate
model.eval()

try:
    t0 = time.time()
    out = generator.generate_mesh(mask_loader=None,clip=args.clip)

    try:
        mesh, stats_dict = out
    except TypeError:
        mesh, stats_dict = out, {}

    mesh_out_file = os.path.join(
        test_out_path, 'mesh_{}{}{}.{}'.format(args.load_iter if args.load_iter else it,
                            '_clip' if args.clip else '',
                            '_obj' if args.obj_only else '',
                            args.mesh_extension))
    mesh.export(mesh_out_file)

except RuntimeError:
    print("Error generating mesh")

