import os
import torch
import torch.nn.functional as F
from .common import (
    get_tensor_values, sample_patch_points, arange_pixels
)
import logging
from .losses import Loss
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
from utils.metrics import MAE
import matplotlib.pyplot as plt
cm = plt.get_cmap('jet')
scale = lambda x : (x-x.min())/(x.max()-x.min())
to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()
to_hw = lambda x, h, w: x.reshape(w,h,-1).permute(1,0,2)

class Trainer(object):
    ''' Trainer object.

    Args:
        model (nn.Module): model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): config file
        device (device): pytorch device
    '''

    def __init__(self, model, optimizer, cfg_all, device=None, **kwargs):
        cfg = cfg_all['training']
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_training_points = cfg['n_training_points']
        self.n_eval_points = cfg['n_training_points']
        self.mask_loss = self.cfg.get('mask_loss',False)

        self.rendering_technique = cfg['type']

        self.loss = Loss(
            cfg['lambda_l1_rgb'], 
            cfg['lambda_normals'],
            cfg.get('lambda_mask', 1.0),
            cfg=cfg,
        )

        self.light_train = self.cfg.get('light_train', False)
        self.amb = self.cfg.get('amb', False)
        self.amb_i = self.cfg.get('amb_i', 0.13)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.train()
        self.optimizer.zero_grad()
        if self.light_train:
            self.light_optimizer.zero_grad()

        loss_dict = self.compute_loss(data, it=it)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        if self.light_train:
            self.light_optimizer.step()
        return loss_dict

    
    def render_visdata(self, data_loader, it, out_render_path):
        save_img = []
        for di, data in enumerate(data_loader):
            if di>=2: break
            (img, mask, world_mat, camera_mat, scale_mat, img_idx, normal, light_src, vis_gt, mask_obj) = \
                self.process_data_dict(data)
            if self.light_train:
                light_src = self.light_para(img_idx.to(self.device))
            h, w = img.shape[-2:] 
            p_loc, pixels = arange_pixels(resolution=(h, w))
            mask_obj_tensor = mask_obj
            pixels = pixels.to(self.device)
            ploc = p_loc.to(self.device)

            img_iter = [to_img(to_numpy(img))[0].transpose(1,2,0)]

            chunk = 4096
            with torch.no_grad():
                rgb_pred, norm_pred, mask_pred, mask_acc, vis_pred, depth_pred = [],[],[],[],[],[]
                rgb_fine, albedo_pred, albedo_fine, spec_pred, spec_fine = [],[],[],[],[]
                for ii, pixels_i in enumerate(torch.split(ploc, chunk, dim=1)):
                    kwargs = {'mask_obj': mask_obj_tensor[0,0].bool().permute(1,0).reshape(1,-1)[:,ii*chunk:min((ii+1)*chunk,ploc.shape[1])]}
                    out_dict = self.model(
                                    pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                                    add_noise=False, eval_=True, it=it, light_src=light_src, **kwargs)
                    rgb_pred.append(out_dict['rgb'])
                    norm_pred.append(out_dict.get('normal_pred',None))
                    mask_pred.append(out_dict.get('mask_pred',None))
                    mask_acc.append(out_dict['acc_map'])
                    depth_pred.append(out_dict['depth'])
                    vis_pred.append(out_dict['vis'])
                    rgb_fine.append(out_dict.get('rgb_fine',None))
                    albedo_pred.append(out_dict['albedo'])
                    spec_pred.append(out_dict['specular'])
                    albedo_fine.append(out_dict.get('albedo_fine',None))
                    spec_fine.append(out_dict.get('specular_fine',None))
        
                rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h,w))
                img_iter.append(to_img(rgb_pred))
                if rgb_fine[0] is not None:
                    rgb_fine = to_numpy(to_hw(torch.cat(rgb_fine, dim=1),h,w))
                    img_iter.append(to_img(rgb_pred))
                if norm_pred[0] is not None:
                    norm_pred = to_numpy(to_hw(torch.cat(norm_pred, dim=1),h,w))
                    norm_pred = np.einsum('ij,hwi->hwj', to_numpy(world_mat)[0,:3,:3]*np.array([[1,-1,-1]]),norm_pred)
                    img_iter.append(to_img(norm_pred/2.+0.5))
                    img_iter.append(to_img(to_numpy(normal[0].permute(1,2,0))/2.+0.5))
                    error = MAE(norm_pred,to_numpy(normal[0].permute(1,2,0)).clip(-1,1))[1]/45
                    img_iter.append(to_img(cm(error.clip(0,1))[...,:3]))
                if mask_pred[0] is not None:
                    mask_pred = to_numpy(to_hw(torch.cat(mask_pred, dim=0),h,w)).repeat(3,axis=-1)
                    img_iter.append(to_img(mask_pred))
                mask_acc = to_numpy(to_hw(torch.cat(mask_acc, dim=0),h,w)).repeat(3,axis=-1)
                img_iter.append(to_img(mask_acc))
                depth_pred = to_numpy(to_hw(torch.cat(depth_pred, dim=0),h,w)).repeat(3,axis=-1)
                depth_pred[mask_pred.astype(bool)] = scale(depth_pred[mask_pred.astype(bool)])                
                img_iter.append(to_img(depth_pred))
                vis_pred = to_numpy(to_hw(torch.cat(vis_pred, dim=0),h,w)).repeat(3,axis=-1)
                img_iter.append(to_img(vis_pred))
                img_iter.append(to_img(to_numpy(vis_gt)[0,0,:,:,None].repeat(3,axis=-1)))
                albedo_pred = to_numpy(to_hw(torch.cat(albedo_pred, dim=1),h,w))
                img_iter.append(to_img(albedo_pred))
                spec_pred = to_numpy(to_hw(torch.cat(spec_pred, dim=1),h,w))
                img_iter.append(to_img(spec_pred))
                if albedo_fine[0] is not None:
                    albedo_fine = to_numpy(to_hw(torch.cat(albedo_fine, dim=1),h,w))
                    img_iter.append(to_img(albedo_fine))
                    spec_fine = to_numpy(to_hw(torch.cat(spec_fine, dim=1),h,w))
                    img_iter.append(to_img(spec_fine))

            with torch.no_grad():
                rgb_pred = \
                    [self.model(
                        pixels_i, camera_mat, world_mat, scale_mat, 'phong_renderer', 
                        add_noise=False, eval_=True, it=it)['rgb']
                        for ii, pixels_i in enumerate(torch.split(ploc, chunk, dim=1))]
           
                rgb_pred = to_numpy(to_hw(torch.cat(rgb_pred, dim=1),h,w))
                img_iter.append(to_img(rgb_pred)) 
            
            save_img.append(np.concatenate(img_iter, axis=-2))
        save_img = np.concatenate(save_img, axis=0)
        save_img = Image.fromarray(save_img.astype(np.uint8)).convert("RGB")
        save_img.save(out_render_path)
        return 

    def process_data_dict(self, data):
        device = self.device

        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        batch_size, _, h, w = img.shape
        mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        normal = data.get('img.normal').to(device)
        light_src = data.get('img.light').to(device)
        vis_img = data.get('img.vis', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        mask_obj = data.get('img.mask_obj', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, normal, light_src, vis_img, mask_obj)

    def compute_loss(self, data, eval_mode=False, it=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        n_points = self.n_eval_points if eval_mode else self.n_training_points
        (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, normal, light_src, vis_gt, mask_obj) = self.process_data_dict(data)
        if self.light_train:
            light_src = self.light_para(img_idx.to(self.device))

        # Shortcuts
        device = self.device
        batch_size, c, h, w = img.shape

        # Assertions
        assert(((h, w) == mask_img.shape[2:4]) and
               (n_points > 0))

        # Sample pixels
        if n_points >= h*w:
            p = arange_pixels((h, w), batch_size)[0].to(device)
            mask_gt = mask_img.bool().reshape(batch_size,-1).to(torch.float32)
            vis_gt = vis_gt.reshape(batch_size,-1).to(torch.float32)
            pix = p
        else:
            p, pix = sample_patch_points(batch_size, n_points,
                                    patch_size=1.,
                                    image_resolution=(h, w),
                                    continuous=False,
                                    )
            p = p.to(device) 
            pix = pix.to(device) 
            mask_gt = get_tensor_values(mask_img, pix.clone()).bool().reshape(batch_size,-1).float()
            mask_obj = get_tensor_values(mask_obj*1., pix.clone()).bool().reshape(batch_size,-1)

        kwargs = {'mask_obj': mask_obj}
        if self.amb:
           kwargs['amb'] = self.amb_i

        out_dict = self.model(
            pix, camera_mat, world_mat, scale_mat, 
            self.rendering_technique, it=it, mask=mask_gt, 
            eval_=eval_mode, light_src=light_src,
            **kwargs,
        )

        rgb_gt = get_tensor_values(img, pix.clone())
        mask_pred = out_dict.get('acc_map',None)
        if not self.mask_loss:
            mask_gt = None
        loss_dict = self.loss(out_dict, rgb_gt.to(device), mask_pred, mask_gt)
        return loss_dict
