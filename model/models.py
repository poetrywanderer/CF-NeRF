'''
Collection of models
'''
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import model.flow.flows as flows

class NeRF_Flows(nn.Module):
    '''
    global learnable parameters for density and rgb, 
    density use conditional tri flow
    rgb use conditional triangular flow
    no prior
    '''
    def __init__(self, args):
        super(NeRF_Flows, self).__init__()
        self.D = args.netdepth
        self.W = args.netwidth
        self.input_ch = args.input_ch
        self.input_ch_views = args.input_ch_views
        self.K_samples = args.K_samples
        self.skips = args.skips
        self.use_viewdirs = args.use_viewdirs
        self.h_alpha_size = args.h_alpha_size
        self.h_rgb_size = args.h_rgb_size
        args.z_size = 3
        self.z_size = args.z_size
        self.n_flows = args.n_flows
        self.type_flows = args.type_flows
        self.n_hidden = args.n_hidden
        self.device = args.device
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W//2)])

        self.alpha_mean = torch.nn.Parameter(torch.zeros(1))
        self.alpha_std = torch.nn.Parameter(torch.ones(1))

        self.rgb_mean = torch.nn.Parameter(torch.zeros(3))
        self.rgb_std = torch.nn.Parameter(torch.ones(3))

        self.intepolation_alpha = torch.empty([2,1]).normal_()
        self.intepolation_rgb = torch.empty([2,3]).normal_()

        self.sample_size = args.K_samples
        self.sample_alpha = torch.empty([self.sample_size,1]).normal_()
        self.sample_rgb = torch.empty([self.sample_size,3]).normal_()

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.W, self.W)
            self.alpha_linear = nn.Linear(self.W, 1)
            self.alpha_std_linear = nn.Linear(self.W, 1)
            self.h_alpha_linear= nn.Linear(self.W, self.h_alpha_size)
            self.h_rgb_linear= nn.Linear(self.W//2, self.h_rgb_size)
        else:
            self.output_linear = nn.Linear(self.W, self.output_ch)
        
        self.flows_rgb = TriangularSylvesterNeRF(args,'rgb')
        self.flows_alpha = TriangularSylvesterNeRF(args,'alpha')
    
    def sample(self, x):
        h_alpha, h_rgb = self.encode(x)
        ######################## entropy loss ########################
        BN = h_alpha.shape[0]
        ## reparameterizate 
        # density
        alpha_mean_k = self.alpha_mean[None,None,:].expand([BN, self.sample_size, 1])
        # alpha_std = F.softplus(self.alpha_std) + 1e-05 
        alpha_std_k = self.alpha_std[None,None,:].expand([BN, self.sample_size, 1])
        eps_alpha = self.sample_alpha[None,...].expand_as(alpha_mean_k)
        alpha0 = eps_alpha.mul(alpha_std_k).add_(alpha_mean_k).view([-1,1]) # (BxNxK,4)
        # rgb
        # rgb_mean_k = self.rgb_mean[None,None,:].expand([BN, self.sample_size, 3])
        # rgb_std = F.softplus(self.rgb_std) + 1e-05 
        # rgb_std_k = rgb_std[None,None,:].expand([BN, self.sample_size, 3])
        # eps_rgb = self.sample_rgb[None,...].expand_as(rgb_mean_k)
        # rgb0 = eps_rgb.mul(rgb_std_k).add_(rgb_mean_k).view([-1,3]) # (BxNxK,4)

        ## pass through flows
        # density
        h_alpha = h_alpha[:,None,:].expand([BN, self.sample_size, self.h_alpha_size])
        h_alpha = h_alpha.reshape([-1,self.h_alpha_size])
        alpha_k, _ = self.flows_alpha(alpha0, h_alpha) # (BxNxK, 1),  (BxNxK,)
        alpha_k = alpha_k.reshape([BN, self.sample_size, 1]) 
        # rgb
        # rgb_k, _ = self.flows_rgb(rgb0, h_rgb) # (BxNxK, 3),  (BxNxK,)

        return alpha_k
    
    def interpolation(self, x, is_val=False):
        h_alpha, h_rgb = self.encode(x)

        ######################## entropy loss ########################
        BN = h_alpha.shape[0] 
        ## reparameterizate 
        # density 
        alpha_mean_k = self.alpha_mean[None,None,:].expand([BN, 2, 1])
        # make sure alpha_std to be positive 
        # alpha_std = F.softplus(self.alpha_std) + 1e-05 
        alpha_std_k = self.alpha_std[None,None,:].expand([BN, 2, 1])
        eps_alpha = self.intepolation_alpha[None,...].expand_as(alpha_mean_k)
        alpha_sample = eps_alpha.mul(alpha_std_k).add_(alpha_mean_k) # (BN, K,1)
        alpha_mean = self.alpha_mean[None,:].expand([BN, 1])
        # # rgb
        rgb_mean_k = self.rgb_mean[None,None,:].expand([BN, 2, 3])
        # make sure alpha_std to be positive
        # rgb_std = F.softplus(self.rgb_std) + 1e-05
        rgb_std_k = self.rgb_std[None,None,:].expand([BN, 2, 3])
        eps_rgb = self.intepolation_rgb[None,...].expand_as(rgb_mean_k)
        rgb_sample = eps_rgb.mul(rgb_std_k).add_(rgb_mean_k) # (BxNxK,3)
        rgb_mean = self.rgb_mean[None,:].expand([BN, 3])

        ## intepolate z1 -> z_mean -> z2 -> z_mean -> z1
        alpha1 = alpha_sample[:,0,:]
        alpha2 = alpha_sample[:,1,:]
        alpha_t = []
        t = list(np.arange(10) / 10.) 
        for beta in t:
            alpha_t.append((1 - beta) * alpha1 + beta * alpha_mean)
        
        for beta in list(np.arange(11) / 10.) :
            alpha_t.append((1 - beta) * alpha_mean + beta * alpha2)
        
        alpha0 = torch.stack(alpha_t,-2).view(-1,1)
        
        ## intepolate z1 -> z_mean -> z2 -> z_mean -> z1
        rgb1 = rgb_sample[:,0,:]
        rgb2 = rgb_sample[:,1,:]
        rgb_t = []
        t = list(np.arange(10) / 10.) 
        for beta in t:
            rgb_t.append((1 - beta) * rgb1 + beta * rgb_mean)
        
        for beta in list(np.arange(11) / 10.) :
            rgb_t.append((1 - beta) * rgb_mean + beta * rgb2)
        
        rgb0 = torch.stack(rgb_t,-2).view(-1,3)

        ## pass through flows
        # density
        h_alpha = h_alpha[:,None,:].expand([BN, len(alpha_t), self.h_alpha_size])
        h_alpha = h_alpha.reshape([-1,self.h_alpha_size])
        z_alpha, _ = self.flows_alpha(alpha0, h_alpha) # (BxNxK, 3),  (BxNxK,)
        z_k_alpha = z_alpha.reshape([BN, len(alpha_t), 1])

        # rgb
        h_rgb = h_rgb[:,None,:].expand([BN, len(rgb_t), self.h_rgb_size])
        h_rgb = h_rgb.reshape([-1,self.h_rgb_size])
        z_rgb, _ = self.flows_rgb(rgb0, h_rgb) # (BxNxK, 3),  (BxNxK,)
        z_k_rgb = z_rgb.reshape([BN, len(rgb_t), 3])

        ###### concate all results 
        z_k_rgb_alpha = torch.cat([z_k_rgb, z_k_alpha], -1) # (BxN, K, 4)

        return z_k_rgb_alpha
    
    def encode(self,x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            h_alpha = self.h_alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
                h_rgb = self.h_rgb_linear(h)
        else:
            h = self.output_linear(h)

        return h_alpha, h_rgb

    def forward(self, x, is_val=False, is_test=False):

        h_alpha, h_rgb = self.encode(x)

        if is_test:
            BN = h_alpha.shape[0] 
            ## reparameterizate 
            # density 
            alpha_mean_k = self.alpha_mean[None,None,:].expand([BN, self.K_samples, 1]) 
            alpha_std_k = self.alpha_std[None,None,:].expand([BN, self.K_samples, 1])
            eps_alpha = self.sample_alpha[None,...].expand_as(alpha_mean_k).to(self.device)
            eps_alpha = torch.cat([eps_alpha[:,:-1,:], torch.zeros([BN,1,1]).to(self.device)],-2)
            alpha0 = eps_alpha.mul(alpha_std_k).add_(alpha_mean_k).view([-1,1]) # (BN, K,1)
            # rgb
            rgb_mean_k = self.rgb_mean[None,None,:].expand([BN, self.K_samples, 3])
            rgb_std_k = self.rgb_std[None,None,:].expand([BN, self.K_samples, 3])
            eps_rgb = self.sample_rgb[None,...].expand_as(rgb_mean_k).to(self.device)
            eps_rgb = torch.cat([eps_rgb[:,:-1,:], torch.zeros([BN,1,3])],-2)
            rgb0 = eps_rgb.mul(rgb_std_k).add_(rgb_mean_k).view([-1,3]) # (BxNxK,3)

            ## pass through flows
            # density
            h_alpha = h_alpha[:,None,:].expand([BN, self.K_samples, self.h_alpha_size])
            h_alpha = h_alpha.reshape([-1,self.h_alpha_size])
            z_alpha, _ = self.flows_alpha(alpha0, h_alpha, is_test) # (BxNxK, 3),  (BxNxK,)
            z_k_alpha = z_alpha.reshape([BN, self.K_samples, 1])
            # rgb
            h_rgb = h_rgb[:,None,:].expand([BN, self.K_samples, self.h_rgb_size])
            h_rgb = h_rgb.reshape([-1,self.h_rgb_size])
            z_rgb, _ = self.flows_rgb(rgb0, h_rgb, is_test) # (BxNxK, 3),  (BxNxK,)
            z_k_rgb = z_rgb.reshape([BN, self.K_samples, 3])

            ###### concate all results 
            z_k_rgb_alpha = torch.cat([z_k_rgb, z_k_alpha], -1) # (BxN, K, 4)

            return z_k_rgb_alpha, torch.zeros_like(z_k_rgb_alpha)

        ######################## entropy loss ########################
        BN = h_alpha.shape[0] 
        ## reparameterizate 
        # density 
        alpha_mean_k = self.alpha_mean[None,None,:].expand([BN, self.K_samples, 1]) 
        # make sure alpha_std to be positive 
        # alpha_std = F.softplus(self.alpha_std) + 1e-05 
        alpha_std_k = self.alpha_std[None,None,:].expand([BN, self.K_samples, 1])
        if not is_test:
            eps_alpha = torch.empty([self.K_samples,1]).normal_()
            eps_alpha = eps_alpha[None,...].expand_as(alpha_mean_k)
        else:
            eps_alpha = self.sample_alpha[None,...].expand_as(alpha_mean_k).to(self.device)
            eps_alpha = torch.cat([eps_alpha[:,:-1,:], torch.zeros([BN,1,1]).to(self.device)],-2)
        alpha0 = eps_alpha.mul(alpha_std_k).add_(alpha_mean_k).view([-1,1]) # (BN, K,1)
        # rgb
        rgb_mean_k = self.rgb_mean[None,None,:].expand([BN, self.K_samples, 3])
        # make sure alpha_std to be positive
        # rgb_std = F.softplus(self.rgb_std) + 1e-05
        rgb_std_k = self.rgb_std[None,None,:].expand([BN, self.K_samples, 3])
        if not is_test:
            eps_rgb = torch.empty([self.K_samples,3]).normal_()
            eps_rgb = eps_rgb[None,...].expand_as(rgb_mean_k)
        else:
            eps_rgb = self.sample_rgb[None,...].expand_as(rgb_mean_k).to(self.device)
            eps_rgb = torch.cat([eps_rgb[:,:-1,:], torch.zeros([BN,1,3])],-2)
        rgb0 = eps_rgb.mul(rgb_std_k).add_(rgb_mean_k).view([-1,3]) # (BxNxK,3)

        ## pass through flows
        # density
        h_alpha = h_alpha[:,None,:].expand([BN, self.K_samples, self.h_alpha_size])
        h_alpha = h_alpha.reshape([-1,self.h_alpha_size])
        z_alpha, sum_log_det_j_alpha = self.flows_alpha(alpha0, h_alpha, is_test) # (BxNxK, 3),  (BxNxK,)
        z_k_alpha = z_alpha.reshape([BN, self.K_samples, 1])
        sum_log_det_j_alpha = sum_log_det_j_alpha.reshape([BN, self.K_samples])

        ## add log_det_jacobian for the last activation functions
        # density softplus
        sum_log_det_j_alpha += z_k_alpha.sum(-1) - F.softplus(z_k_alpha).sum(-1)

        ## compute loss_entropy
        # density
        alpha0 = alpha0.reshape([BN, self.K_samples, 1])
        base_log_norm_alpha = -0.5 * (alpha_std_k.log()*2 + (alpha0 - alpha_mean_k) * (alpha0 - alpha_mean_k) * (alpha_std_k**2).reciprocal())

        # rgb
        h_rgb = h_rgb[:,None,:].expand([BN, self.K_samples, self.h_rgb_size])
        h_rgb = h_rgb.reshape([-1,self.h_rgb_size])
        z_rgb, sum_log_det_j_rgb = self.flows_rgb(rgb0, h_rgb, is_test) # (BxNxK, 3),  (BxNxK,)
        z_k_rgb = z_rgb.reshape([BN, self.K_samples, 3])
        sum_log_det_j_rgb = sum_log_det_j_rgb.reshape([BN, self.K_samples])

        # rgb
        sum_log_det_j_rgb += z_k_rgb.sum(-1) - 2 * F.softplus(z_k_rgb).sum(-1)

        ## compute loss_entropy
        # rgb
        rgb0 = rgb0.reshape([BN, self.K_samples, 3]) 
        base_log_norm_rgb = -0.5 * (rgb_std_k.log()*2 + (rgb0 - rgb_mean_k) * (rgb0 - rgb_mean_k) * (rgb_std_k**2).reciprocal())

        ## sum up loss_entropy
        loss_entropy = base_log_norm_alpha.mean() - sum_log_det_j_alpha.mean() + base_log_norm_rgb.mean() - sum_log_det_j_rgb.mean()

        ###### concate all results 
        z_k_rgb_alpha = torch.cat([z_k_rgb, z_k_alpha], -1) # (BxN, K, 4)

        return z_k_rgb_alpha, loss_entropy.expand_as(z_k_alpha)


class TriangularSylvesterNeRF(nn.Module):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, args, flag='1'):
        super(TriangularSylvesterNeRF, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.TriangularSylvester
        if flag == 'alpha':
            self.z_size = 1
            self.num_flows = args.n_flows
            # self.num_flows = 1
            self.q_z_nn_output_dim = args.h_alpha_size
        elif flag == 'rgb':
            self.z_size = 3
            self.num_flows = args.n_flows
            self.q_z_nn_output_dim = args.h_rgb_size
        else:
            self.z_size = args.z_size
            self.num_flows = args.n_flows
            self.q_z_nn_output_dim = args.h_size

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
            self.diag_activation
        )

        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module('flow_' + str(k), flow_k)

    def encode(self, h):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = h.size(0)

        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)

        full_d = full_d.reshape(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.reshape(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.reshape(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(h)

        # reshape flow parameters to divide over K flows
        b = b.reshape(batch_size, 1, self.z_size, self.num_flows)

        return r1, r2, b

    def forward(self, z0, h, is_test):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        r1, r2, b = self.encode(h)

        # Sample z_0
        z = [z0]

        # Normalizing flows
        for k in range(self.num_flows):

            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True,is_test=is_test)

            z.append(z_k)
            self.log_det_j += log_det_jacobian


        return z[-1], self.log_det_j