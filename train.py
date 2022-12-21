import os
import sys
import logging
import time
import shutil, _pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl

logger_py = logging.getLogger(__name__)

np.random.seed(42)
torch.manual_seed(42)
from utils.tools import set_debugger
set_debugger()

# Arguments
parser = argparse.ArgumentParser(
    description='Training of UNISURF model'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--gpu', type=int, help='gpu')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds with exit code 2.')

args = parser.parse_args()
cfg = dl.load_config(args.config, )
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# params
out_dir = cfg['training']['out_dir']
backup_every = cfg['training']['backup_every']
lr = cfg['training']['learning_rate']
exit_after = args.exit_after
expname = cfg['training']['out_dir'].rstrip('/').split('/')[-1]

os.makedirs(out_dir, exist_ok=True)
if args.config is not None:
    f = os.path.join(out_dir, 'config.yaml')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())

# init dataloader
train_loader = dl.get_dataloader(cfg, mode='train')
test_loader = dl.get_dataloader(cfg, mode='test')

# init network
model = mdl.NeuralNetwork(cfg)
print(model)
# init renderer
renderer = mdl.Renderer(model, cfg, device=device)
# init optimizer
weight_decay = cfg['training']['weight_decay']
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# init training
trainer = mdl.Trainer(renderer, optimizer, cfg, device=device)

kwargs = {}
if trainer.light_train:
    light_opt = train_loader.dataset.light_direction.copy()
    trainer.light_para = torch.nn.Embedding(light_opt.shape[0], 3, sparse=True).to(device)
    trainer.light_para.weight.data.copy_(torch.tensor(light_opt.copy()))
    light_para_list = [{'params':list(trainer.light_para.parameters())}]
    trainer.light_optimizer = torch.optim.SparseAdam(light_para_list, lr=cfg['training'].get('lr_light',0.001))
    kwargs['light_para'] = trainer.light_para
    kwargs['light_optimizer'] = trainer.light_optimizer

# init checkpoints and load
checkpoint_io = mdl.CheckpointIO(os.path.join(out_dir,'models'), model=model, optimizer=optimizer, **kwargs,)

try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, cfg['training']['scheduler_milestones'],
    gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

if trainer.light_train:
    scheduler_light = optim.lr_scheduler.MultiStepLR(
        trainer.light_optimizer, cfg['training']['scheduler_light'],
        gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)
    # save initialized light para
    if it == -1:
        with open(os.path.join(out_dir, 'light_para.pd'),'wb') as f0:
            _pickle.dump({0:light_opt.astype(np.float32)}, f0)

logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    
# init training output
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
visualize_every = cfg['training']['visualize_every']
if visualize_every > 0:
    visualize_path = os.path.join(out_dir, 'images')
    os.makedirs(visualize_path, exist_ok=True)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info(model)
logger_py.info('Total number of parameters: %d' % nparameters)
t0b = time.time()

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss_dict = trainer.train_step(batch, it)
        loss = loss_dict['loss']
        metric_val_best = loss
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            plog = '[%s][%s] [Epoch %02d] it=%03d, time=%.4f' \
                           % (cfg['dataloading']['obj_name'],expname,epoch_it, it, time.time() - t0b)
            if trainer.light_train:
                light_error = ((trainer.light_para.weight.data.detach() - torch.tensor(light_opt).to(device))**2).mean()
                plog += ', light_error=%.6f' % (light_error.detach().cpu())
                logger.add_scalar('train/light_error', light_error.detach().cpu(), it)
            for l, num in loss_dict.items():
                logger.add_scalar('train/'+l, num.detach().cpu(), it)
                plog += f', {l}={num.detach().cpu():.4f}'
            print(plog)
            logger_py.info(plog)
            t0b = time.time()
        
        if visualize_every > 0 and (it % visualize_every==0 or it in [gi*1000 for gi in range(10)]+[gi*200 for gi in range(5)]):
            logger_py.info("Rendering")
            out_render_path = os.path.join(visualize_path, 'vis_{:06d}.png'.format(it))
            trainer.render_visdata(
                        test_loader, 
                        it, out_render_path)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # save light parameters
        if trainer.light_train and it>0 and (it % 5000 == 0 or it in [gi*1000 for gi in range(10)]+[gi*200 for gi in range(5)]):
            with open(os.path.join(out_dir, 'light_para.pd'),'rb') as f0:
                light_saved = _pickle.load(f0)
            light_saved[it] = trainer.light_para.weight.data.detach().cpu().numpy()
            with open(os.path.join(out_dir, 'light_para.pd'),'wb') as f0:
                _pickle.dump(light_saved, f0)

    scheduler.step()
    if trainer.light_train:
        scheduler_light.step()