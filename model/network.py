import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field
    
    Args:
        cfg (dict): network configs
    '''
    def __init__(self, cfg_all, **kwargs):
        super().__init__()
        cfg = cfg_all['model']
        out_dim = 4
        dim = 3
        self.num_layers = cfg['num_layers']
        hidden_size = cfg['hidden_dim']
        self.octaves_pe = cfg['octaves_pe']
        self.octaves_pe_views = cfg['octaves_pe_views']
        self.skips = cfg['skips']
        self.rescale = cfg['rescale']
        self.feat_size = cfg['feat_size']
        geometric_init = cfg['geometric_init'] 
        self.light_int_src = cfg.get('light_int', 4)
        self.env = False
        self.edit_mask = None
        self.albedo_new = None
        self.basis_new = None

        bias = 0.6

        # init pe
        dim_embed = dim*self.octaves_pe*2 + dim
        dim_embed_view = dim + dim*self.octaves_pe*2 + self.feat_size 
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)
        self.sgbasis = SGBasis(nbasis=9)

        ### geo network
        dims_geo = [dim_embed]+ [ hidden_size if i in self.skips else hidden_size for i in range(0, self.num_layers)] + [self.feat_size+1] 
        self.num_layers = len(dims_geo)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]

            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

        ## appearance network
        dims_view = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3]
        dims_spec = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3*9]

        self.num_layers_app = len(dims_view)
        self.num_layers_spec = len(dims_spec)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        for l in range(0, self.num_layers_spec - 1):
            out_dim = dims_spec[l + 1]
            lina = nn.Linear(dims_spec[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina_spec" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occ(self, p):
        pe = self.transform_points(p/self.rescale)
        x = pe
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
 
    def infer_app(self, points, normals, view_dirs, feature_vectors, **kwargs):
        app_input = torch.cat([self.transform_points(points/self.rescale), feature_vectors], dim=-1)
        x = app_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        x = self.tanh(x) * 0.5 + 0.5
        y = app_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina_spec" + str(l))
            y = lina(y)
            if l < self.num_layers_app - 2:
                y = self.relu(y)
        y = self.relu(y)
        pts2l = kwargs['pts2l']
        pts2c = view_dirs
        if self.albedo_new is not None:
            x[self.edit_mask] = self.albedo_new
        if self.basis_new is not None:
            weight_new = torch.zeros_like(y)
            curr_basis = np.bincount(y.detach().cpu().numpy().reshape(-1,9).argmax(axis=-1)).argmax()
            weight_new.view(-1,3,9)[:,:,self.basis_new][self.edit_mask]=2**self.basis_new/10
            y = weight_new.reshape(-1,3*9)
        brdf, rough = self.sgbasis(l=pts2l, v=pts2c, n=normals,albedo=x, weights=y)
        cos = torch.einsum('ni,ni->n', pts2l, normals).reshape(-1,1)
        light_int = self.light_int_src / kwargs['pts2l_dis'][...,None]**2
        
        rgb = brdf*light_int*cos
        rgb = rgb.clamp(0,1) if not self.env else rgb
        return rgb, x, rough, y

    def gradient(self, p, tflag=True):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_occ(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=tflag,
                retain_graph=tflag,
                only_inputs=True, allow_unused=tflag)[0]
            return gradients.unsqueeze(1)

    def forward(self, p, ray_d=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        x = self.infer_occ(p)
        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif ray_d is not None:
            input_views = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            normals =  self.gradient(p).squeeze(1)
            normals = F.normalize(normals,dim=-1)
            rgb, albedo, spec, weights = self.infer_app(p, normals, input_views, x[...,1:], **kwargs)
            if return_addocc:
                return rgb, self.sigmoid(x[...,:1] * -10.0 ), albedo, spec, weights
            else:
                return rgb, albedo, spec, weights
        elif return_logits:
            return -1*x[...,:1]


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)


class SGBasis(nn.Module):
    def __init__(self, nbasis=9, specular_rgb=True):
        super().__init__()
        self.nbasis = nbasis
        self.specular_rgb = specular_rgb
        self.lobe = nn.Parameter(torch.tensor([np.exp(i) for i in range(2,11)],dtype=torch.float32))
        self.lobe.requires_grad_(False)

    def forward(self, v, n, l, albedo, weights):
        '''
        :param  v: [N, 3]
        :param  n: [N, 3]
        :param  l: [N, 3]
        :param  albedo: [N, 3]
        :param  weights: [N, nbasis]
        '''
        h = F.normalize(l + v, dim=-1)   # [N,3]
        D = torch.exp(self.lobe[None,].clamp(min=0) * ((h*n).sum(-1, keepdim=True) - 1))  # [N, 9]
        if self.specular_rgb:
            specular = (weights.view(-1,3,self.nbasis) * D[:,None]).sum(-1).clamp(min=0.)  # [N,]
        else:
            specular = (weights * D).sum(-1, keepdim=True).clamp(min=0.)  # [N,]
        lambert = albedo 
        brdf = lambert + specular.expand_as(lambert)
        return brdf, specular