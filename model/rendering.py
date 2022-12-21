import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import (
    get_mask, origin_to_world,image_points_to_ray)

epsilon = 1e-6
class Renderer(nn.Module):
    ''' Renderer class containing unisurf and phong rendering(adapted from IDR)
    
    Args:
        model (nn.Module): model
        cfg (dict): network configs
    '''

    def __init__(self, model, cfg_all, device=None, **kwargs):
        super().__init__()
        cfg = cfg_all['rendering']
        self.cfg=cfg
        self._device = device
        self.depth_range = [cfg['near'], cfg['far']]
        self.n_max_network_queries = cfg['n_max_network_queries']
        self.white_background = cfg['white_background']
        self.model = model.to(device)
        
        self.vis_after = self.cfg.get('vis_after',False)
        self.vis_after_iter = self.cfg.get('vis_after_iter', 1000)
        self.render_fine = self.cfg.get('render_fine', False)
        self.render_fine_iter = self.cfg.get('render_fine_iter', -1)
        self.vis_steps = self.cfg.get('vis_steps', 128)
        self.no_vis = self.cfg.get('no_vis', False)
        self.im_res = None
        self.env = False
        self.light_int_src = None
        self.edit_mask = None


    def forward(self, pixels, camera_mat, world_mat, scale_mat, 
                      rendering_technique, add_noise=True, eval_=False,
                      it=0, light_src=None, novel_view=False, view_ori=None,
                      vis_pre=None, **kwargs):
        if rendering_technique == 'unisurf':
            out_dict = self.unisurf(
                pixels, camera_mat, world_mat, scale_mat, 
                light_src=light_src, it=it, add_noise=add_noise, eval_=eval_,
                novel_view=novel_view, view_ori=view_ori, vis_pre=vis_pre, **kwargs,
            )
        elif rendering_technique == 'phong_renderer':
            out_dict = self.phong_renderer(
                pixels, camera_mat, world_mat, scale_mat
            )
        else:
            print("Choose unisurf or phong_renderer")
        return out_dict
        
    def unisurf(self, pixels, camera_mat, world_mat, 
                scale_mat, light_src=None, add_noise=False, it=0, eval_=False,
                novel_view=False, view_ori=None, vis_pre=None, **kwargs):
        # Get configs
        batch_size, n_points, _ = pixels.shape
        device = self._device
        rad = self.cfg['radius']
        ada_start = self.cfg['interval_start']
        ada_end = self.cfg['interval_end']
        ada_grad = self.cfg['interval_decay']
        steps = self.cfg['num_points_in']
        steps_outside = self.cfg['num_points_out']
        ray_steps = self.cfg['ray_marching_steps']

        depth_range = torch.tensor(self.depth_range)
        n_max_network_queries = self.n_max_network_queries
        
        camera_world = origin_to_world(
            n_points, camera_mat, world_mat, scale_mat
        )
        ray_vector = image_points_to_ray(pixels, camera_mat, world_mat)
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)
 
        # Get sphere intersection
        depth_intersect,_ = get_sphere_intersection(
            camera_world[:,0], ray_vector, r=rad
        )
        
        # Find surface
        with torch.no_grad():
            d_i = self.ray_marching(
                camera_world, ray_vector, self.model,
                n_secant_steps=8, 
                n_steps=[int(ray_steps),int(ray_steps)+1], 
                rad=rad,
                depth_range=self.depth_range,
            )

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0
        d_i = d_i.detach()

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()
        if novel_view:
            h, w = self.im_res
            p_corner = torch.tensor([[
                [0,0], [w-1,0], [w-1,h-1], [0,h-1]
            ]],dtype=torch.float32).to(device)
            camloc_ori = origin_to_world(
                1, camera_mat, view_ori, scale_mat
            )
            ray_corner = image_points_to_ray(p_corner, camera_mat, view_ori).view(-1,3)
            ray_corner = F.normalize(ray_corner,dim=-1)
            ray_corner_2 = torch.cat([ray_corner[1:],ray_corner[:1]],dim=0)
            n_plane = F.normalize(torch.cross(ray_corner,ray_corner_2))
            psuf = camera_world + ray_vector * d_i.unsqueeze(-1)
            vsuf = F.normalize(psuf - camloc_ori, dim=-1).view(-1,1,3)
            pdot = (vsuf*n_plane).sum(-1)
            mask_room = (pdot>=0).all(-1)
            mask_pred = mask_pred & mask_room

        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device) 
            dists *= depth_range[1]
            dists[mask_pred] = d_i[mask_pred]
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]

        # Project depth to 3d poinsts
        camera_world = camera_world.reshape(-1, 3)
        ray_vector = ray_vector.reshape(-1, 3)
       
        points = camera_world + ray_vector * dists.unsqueeze(-1)
        points = points.view(-1,3)

        # Define interval
        depth_intersect[:,:,0] = torch.Tensor([0.0]).cuda() 
        dists_intersect = depth_intersect.reshape(-1, 2)

        d_inter = dists[network_object_mask]
        d_sphere_surf = dists_intersect[network_object_mask][:,1]
        delta = torch.max(ada_start * torch.exp(-1 * ada_grad * it * torch.ones(1)),\
             ada_end * torch.ones(1)).cuda()

        dnp = d_inter - delta
        dfp = d_inter + delta
        dnp = torch.where(dnp < depth_range[0].float().to(device),\
            depth_range[0].float().to(device), dnp)
        dfp = torch.where(dfp >  d_sphere_surf,  d_sphere_surf, dfp)
        if (dnp!=0.0).all() and it > 5000:
            full_steps = steps+steps_outside
        else:
            full_steps = steps

        d_nointer = dists_intersect[~network_object_mask]

        d2 = torch.linspace(0., 1., steps=full_steps, device=device)
        d2 = d2.view(1, 1, -1).repeat(batch_size, d_nointer.shape[0], 1)
        d2 = depth_range[0] * (1. - d2) + d_nointer[:,1].view(1, -1, 1)* d2

        if add_noise:
            di_mid = .5 * (d2[:, :, 1:] + d2[:, :, :-1])
            di_high = torch.cat([di_mid, d2[:, :, -1:]], dim=-1)
            di_low = torch.cat([d2[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(batch_size, d2.shape[1], full_steps, device=device)
            d2 = di_low + (di_high - di_low) * noise 
        
        p_noiter = camera_world[~network_object_mask].unsqueeze(-2) \
            + ray_vector[~network_object_mask].unsqueeze(-2) * d2.unsqueeze(-1)
        p_noiter = p_noiter.reshape(-1, 3)

        # Sampling region with surface intersection        
        d_interval = torch.linspace(0., 1., steps=steps, device=device)
        d_interval = d_interval.view(1, 1, -1).repeat(batch_size, d_inter.shape[0], 1)        
        d_interval = (dnp).view(1, -1, 1) * (1. - d_interval) + (dfp).view(1, -1, 1) * d_interval

        if full_steps != steps:
            d_binterval = torch.linspace(0., 1., steps=steps_outside, device=device)
            d_binterval = d_binterval.view(1, 1, -1).repeat(batch_size, d_inter.shape[0], 1)
            d_binterval =  depth_range[0] * (1. - d_binterval) + (dnp).view(1, -1, 1)* d_binterval
            d1,_ = torch.sort(torch.cat([d_binterval, d_interval],dim=-1), dim=-1)
        else:
            d1 = d_interval

        if add_noise:
            di_mid = .5 * (d1[:, :, 1:] + d1[:, :, :-1])
            di_high = torch.cat([di_mid, d1[:, :, -1:]], dim=-1)
            di_low = torch.cat([d1[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(batch_size, d1.shape[1], full_steps, device=device)
            d1 = di_low + (di_high - di_low) * noise 

        p_iter = camera_world[network_object_mask].unsqueeze(-2)\
             + ray_vector[network_object_mask].unsqueeze(-2) * d1.unsqueeze(-1)
        p_iter = p_iter.reshape(-1, 3)

        # Merge rendering points
        p_fg = torch.zeros(batch_size * n_points, full_steps, 3, device=device)
        p_fg[~network_object_mask] =  p_noiter.view(-1, full_steps,3)
        p_fg[network_object_mask] =  p_iter.view(-1, full_steps,3)
        p_fg = p_fg.reshape(-1, 3)
        ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
        ray_vector_fg = -1*ray_vector_fg.reshape(-1, 3)

        surface_mask = network_object_mask.view(-1)
        surface_points = points[surface_mask]
        N = surface_points.shape[0]
        surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01      
        pp = torch.cat([surface_points, surface_points_neig], dim=0)
        g = self.model.gradient(pp) 
        normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 10**(-5))
        diff_norm =  torch.norm(normals_[:N] - normals_[N:], dim=-1)
        normal = torch.zeros_like(p_fg.reshape(batch_size * n_points, full_steps, 3)[:,0])
        normal[surface_mask] = normals_[:N]

        if self.env:
            rgb_fine_all, vis_pred_all = [],[]
            for lidx, light_i in enumerate(light_src):
                self.model.light_int_src = self.light_int_src[lidx]
                light_i = light_i[None,]
                pts2l_dis = light_i.repeat(p_fg.shape[0],1) - p_fg
                pts2l = F.normalize(pts2l_dis,dim=-1)
                
                # visibility
                if vis_pre is None:
                    if self.no_vis:
                        vis = torch.ones_like(points[...,0])
                    else:
                        if self.vis_after and it<self.vis_after_iter:
                            vis = torch.ones_like(points[...,0])
                        else:
                            lnear = 0.1
                            pts2l_dis_vis = light_i.repeat(points.shape[0],1) - points
                            pts2l_dis_cal = light_i.repeat(points.shape[0],1)/100*3 - points
                            pts2l_vis = F.normalize(pts2l_dis_vis,dim=-1)
                            vis = self.light_visibility(surf=points, light_dir=pts2l_vis,
                                        lnear=lnear, 
                                        lfar=torch.norm(pts2l_dis_cal,dim=-1),
                                        n_steps=self.vis_steps, max_points=3500000,
                                        )
                            vis[~network_object_mask.view(-1)] = 1
                else:
                    vis = vis_pre[lidx:lidx+1]
                rgb_fine = torch.ones_like(points) if self.white_background else torch.zeros_like(points)
                rgb_mask = []
                ray_vector_mask = -1*ray_vector.view(-1,3)[surface_mask]
                pts2l_dis_mask = light_i.repeat(surface_points.shape[0],1) - surface_points
                pts2l_mask = F.normalize(pts2l_dis_mask,dim=-1)
                for i in range(0, surface_points.shape[0], n_max_network_queries):
                    rgb_i, _, _, _, _ = self.model(
                        surface_points[i:i+n_max_network_queries], 
                        ray_vector_mask[i:i+n_max_network_queries], 
                        return_addocc=True, noise=False,
                        pts2l=pts2l_mask[i:i+n_max_network_queries],
                        pts2l_dis=torch.norm(pts2l_dis_mask,dim=-1)[i:i+n_max_network_queries],
                    )
                    rgb_mask.append(rgb_i)
                if surface_points.shape[0]>0:
                    rgb_fine[surface_mask]  = torch.cat(rgb_mask, dim=0).reshape(-1,3)
                rgb_fine_all.append(rgb_fine * vis[...,None])
                vis_pred_all.append(vis)
            rgb_fine = torch.stack(rgb_fine_all,dim=0).sum(0)
            vis_all = torch.stack(vis_pred_all,dim=0)
            vis = vis_all.mean(0)
                    
            out_dict = {
                'vis' : vis.reshape(batch_size, -1),
                'vis_all' : vis_all.reshape(light_src.shape[0], -1) if vis_pre is None else None,
                'rgb_fine': rgb_fine.reshape(batch_size,-1,3),
            }
            return out_dict


        pts2l_dis = light_src.repeat(p_fg.shape[0],1) - p_fg
        pts2l = F.normalize(pts2l_dis,dim=-1)

        # visibility
        if self.no_vis or (self.vis_after and it<self.vis_after_iter):
            vis = torch.ones_like(points[...,0])
        else:
            lnear = 0.3
            if it>5000:
                lnear = 0.2
            if it>30000 or eval_:
                lnear = 0.1
            pts2l_dis_vis = light_src.repeat(points.shape[0],1) - points
            pts2l_vis = F.normalize(pts2l_dis_vis,dim=-1)
            vis = self.light_visibility(surf=points, light_dir=pts2l_vis,
                            lnear=lnear, 
                            lfar=torch.norm(pts2l_dis_vis,dim=-1),
                            n_steps=self.vis_steps, max_points=3500000,
                            )
            vis[~network_object_mask.view(-1)] = 1

        noise = not eval_
        rgb_fg, logits_alpha_fg = [], []
        albedo_fg, spec_fg = [],[]
        for i in range(0, p_fg.shape[0], n_max_network_queries):
            if self.edit_mask is not None:
                self.model.edit_mask = self.edit_mask[:,None].repeat(1,full_steps).view(-1)[i:i+n_max_network_queries]
            rgb_i, logits_alpha_i, albedo_i, spec_i, _ = self.model(
                p_fg[i:i+n_max_network_queries], 
                ray_vector_fg[i:i+n_max_network_queries], 
                return_addocc=True, noise=noise,
                pts2l=pts2l[i:i+n_max_network_queries], vis=vis[i:i+n_max_network_queries], 
                normal = normal.unsqueeze(1).repeat(1,full_steps,1).view(-1,3)[i:i+n_max_network_queries],
                pts2l_dis=torch.norm(pts2l_dis,dim=-1)[i:i+n_max_network_queries],
            )
            rgb_fg.append(rgb_i)
            logits_alpha_fg.append(logits_alpha_i)
            albedo_fg.append(albedo_i)
            spec_fg.append(spec_i)

        rgb_fg = torch.cat(rgb_fg, dim=0)
        logits_alpha_fg = torch.cat(logits_alpha_fg, dim=0)
        
        rgb = rgb_fg.reshape(batch_size * n_points, full_steps, 3)
        alpha = logits_alpha_fg.view(batch_size * n_points, full_steps)

        weights = alpha * torch.cumprod(torch.cat([torch.ones((rgb.shape[0], 1), device=device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        rgb_values = (rgb_values * vis[...,None]).clamp(0,1) if not self.env else rgb_values * vis[...,None]

        albedo_fg = torch.cat(albedo_fg, dim=0).reshape(batch_size * n_points, full_steps, 3)
        spec_fg = torch.cat(spec_fg, dim=0).reshape(batch_size * n_points, full_steps, 3)
        albedo_values = torch.sum(weights.unsqueeze(-1) * albedo_fg, dim=-2)
        spec_values = torch.sum(weights.unsqueeze(-1) * spec_fg, dim=-2)
        
        acc_map = torch.sum(weights, -1)
        if 'amb' in kwargs:
            rgb_values = (rgb_values + kwargs['amb'] * albedo_values).clamp(0,1)
        
        if self.white_background:
            rgb_values = rgb_values + (1. - acc_map.unsqueeze(-1))

        depth =torch.ones_like(d_i).to(device)
        depth[mask_pred] = d_i[mask_pred]

        if self.render_fine and it> self.render_fine_iter:
            rgb_fine = torch.ones_like(points) if self.white_background else torch.zeros_like(points)
            albedo_fine = torch.ones_like(points)
            spec_fine = torch.ones_like(points)
            rgb_mask, albedo_mask, spec_mask = [],[],[]
            ray_vector_mask = -1*ray_vector.view(-1,3)[surface_mask]
            pts2l_dis_mask = light_src.repeat(surface_points.shape[0],1) - surface_points
            pts2l_mask = F.normalize(pts2l_dis_mask,dim=-1)
            for i in range(0, surface_points.shape[0], n_max_network_queries):
                if self.edit_mask is not None:
                    self.model.edit_mask = self.edit_mask[surface_mask][i:i+n_max_network_queries]
                rgb_i, logits_alpha_i, albedo_i, spec_i, _ = self.model(
                    surface_points[i:i+n_max_network_queries], 
                    ray_vector_mask[i:i+n_max_network_queries], 
                    return_addocc=True, noise=noise,
                    pts2l=pts2l_mask[i:i+n_max_network_queries],
                    pts2l_dis=torch.norm(pts2l_dis_mask,dim=-1)[i:i+n_max_network_queries],
                )
                rgb_mask.append(rgb_i)
                albedo_mask.append(albedo_i)
                spec_mask.append(spec_i)
            if surface_points.shape[0]>0:
                rgb_fine[surface_mask]  = torch.cat(rgb_mask, dim=0).reshape(-1,3)
                albedo_fine[surface_mask] = torch.cat(albedo_mask, dim=0).reshape(-1, 3)
                spec_fine[surface_mask] = torch.cat(spec_mask, dim=0).reshape(-1, 3)
            rgb_fine = (rgb_fine * vis[...,None]).clamp(0,1) if not self.env else rgb_fine * vis[...,None]
            if surface_points.shape[0]>0 and 'amb' in kwargs:
                rgb_fine[surface_mask] = (rgb_fine[surface_mask] + kwargs['amb'] * albedo_fine[surface_mask]).clamp(0,1)
                
        
        out_dict = {
            'rgb': rgb_values.reshape(batch_size, -1, 3),
            'mask_pred': network_object_mask,
            'diff_norm': diff_norm,
            'normal_pred': normal.reshape(batch_size, -1, 3),
            'acc_map' : acc_map.reshape(batch_size, -1),
            'depth' : depth.reshape(batch_size,-1),
            'vis' : vis.reshape(batch_size, -1),
            'albedo': albedo_values.reshape(batch_size, -1, 3),
            'specular': spec_values.reshape(batch_size, -1, 3),
            'rgb_fine': rgb_fine.reshape(batch_size,-1,3) if self.render_fine and it> self.render_fine_iter else None,
            'albedo_fine': albedo_fine.reshape(batch_size, -1, 3) if self.render_fine and it> self.render_fine_iter else None,
            'specular_fine': spec_fine.reshape(batch_size, -1, 3) if self.render_fine and it> self.render_fine_iter else None,
        }
        
        return out_dict

    def phong_renderer(self, pixels, camera_mat, world_mat, 
                     scale_mat):
        batch_size, num_pixels, _ = pixels.shape
        device = self._device
        rad = self.cfg['radius']
        n_points = num_pixels
        camera_world = origin_to_world(num_pixels, camera_mat, world_mat, scale_mat)
        ray_vector = image_points_to_ray(pixels, camera_mat, world_mat)
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)

        light_source = camera_world[0,0] 
        light = (light_source / light_source.norm(2)).unsqueeze(1).cuda()
    
        diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
        ambiant = torch.Tensor([0.3,0.3,0.3]).float()

        # run ray tracer / depth function --> 3D point on surface (differentiable)
        self.model.eval()
        with torch.no_grad():
            d_i = self.ray_marching(camera_world, ray_vector, self.model,
                                         n_secant_steps=8,  n_steps=[int(512),int(512)+1], rad=rad,
                                        depth_range=self.depth_range,)
        # Get mask for where first evaluation point is occupied
        d_i = d_i.detach()
    
        mask_zero_occupied = d_i == 0
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device)
            dists[mask_pred] = d_i[mask_pred].detach()
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]

            camera_world = camera_world.reshape(-1, 3)
            ray_vector = ray_vector.reshape(-1, 3)

            points = camera_world + ray_vector * dists.unsqueeze(-1)
            points = points.view(-1,3)
            view_vol = -1 * ray_vector.view(-1, 3)
            rgb_values = torch.ones_like(points).float().cuda()

            surface_points = points[network_object_mask]
            surface_view_vol = view_vol[network_object_mask]

            # Derive Normals
            grad = []
            for pnts in torch.split(surface_points, 1000000, dim=0):
                grad.append(self.model.gradient(pnts)[:,0,:].detach())
                torch.cuda.empty_cache()
            grad = torch.cat(grad,0)
            surface_normals = grad / grad.norm(2,1,keepdim=True)

        diffuse = torch.mm(surface_normals, light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0).cuda()
        rgb_values[network_object_mask] = (ambiant.unsqueeze(0).cuda() + diffuse).clamp_max(1.0)

        out_dict = {
            'rgb': rgb_values.reshape(batch_size, -1, 3),
        }

        return out_dict

    
    def light_visibility(self, surf=None, light_dir=None,
                        lnear=0.1, lfar=3.5,
                        n_steps=128, max_points=3500000,
                        ):
        rays_o = surf
        rays_d = light_dir
        device = surf.device

        d_proposal = torch.linspace(0, 1, steps=n_steps).view(1, n_steps, 1).to(device)
        d_proposal = lnear * (1. - d_proposal) + lfar.view(-1,1,1).repeat(1, n_steps, 1)* d_proposal
        p_proposal = rays_o.unsqueeze(-2).repeat(1, n_steps, 1) + \
                        rays_d.unsqueeze(-2).repeat(1, n_steps, 1) * d_proposal
        # [L,N,S,3]
        with torch.enable_grad():
            alpha = torch.cat([
                self.model(p_split, only_occupancy=True) 
                for p_split in torch.split(
                    p_proposal.reshape(1, -1, 3),
                    int(max_points), dim=1)], dim=1).view(-1, n_steps)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        acc_map = torch.sum(weights, -1)

        return 1 - acc_map

    def ray_marching(self, ray0, ray_direction, model, c=None,
                             tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                             depth_range=[25,40], max_points=3500000, rad=1.0,
                             ):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        tau = 0.5
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

            
        depth_intersect, _ = get_sphere_intersection(ray0[:,0], ray_direction, r=rad)
        d_intersect = depth_intersect[...,1]            
        
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(1, -1, 1,1)* d_proposal

        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

        # Evaluate all proposal points in parallel
        with torch.no_grad():
            val = torch.cat([(
                self.model(p_split, only_occupancy=True) - tau)
                for p_split in torch.split(
                    p_proposal.reshape(batch_size, -1, 3),
                    int(max_points / batch_size), dim=1)], dim=1).view(
                        batch_size, -1, n_steps)

        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension                    [B,N,S]
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)     # -n for min cost

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        # get X_n, X_n+1 and density of them
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = self.secant(
            f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
            ray_direction_masked, tau)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out

    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, it=0):
        ''' Runs the secant method for interval [d_low, d_high].

        Args:
            d_low (tensor): start values for the interval
            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                f_mid = self.model(p_mid,  batchwise=False,
                                only_occupancy=True, it=it)[...,0] - tau
            ind_low = f_mid < 0
            ind_low = ind_low
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def transform_to_homogenous(self, p):
        device = self._device
        batch_size, num_points, _ = p.size()
        r = torch.sqrt(torch.sum(p**2, dim=2, keepdim=True))
        p_homo = torch.cat((p, torch.ones(batch_size, num_points, 1).to(device)), dim=2) / r
        return p_homo


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect