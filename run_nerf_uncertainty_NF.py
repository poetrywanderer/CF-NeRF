'''
faster training using strategies from dsnerf + softplus rule
'''
from ast import In
from operator import truediv
import os, sys
from pickle import TRUE
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import cv2
import math

from skimage.metrics import structural_similarity as SSIM

from visualization_funcs import *

# import torch.distributions as D

# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import * 
from load_blender import load_blender_data
from model.models import *

from collections import OrderedDict

from scipy import stats

# from plot_snippets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False 

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs, is_val, is_test):
        for i in range(0, inputs.shape[0], chunk):
            t0 = time.time()
            a, b = fn(inputs[i:i+chunk], is_val, is_test)
            # print('single forward time:',time.time()-t0)
            if i == 0:
                A = a
                B = b
            else:
                A = torch.cat([A,a],dim=0)
                B = torch.cat([B,b],dim=0)
        return A, B
    return ret


def run_network(inputs, viewdirs, fn, is_val, is_test, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        if viewdirs.ndim == inputs.ndim -1:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
        else:
            input_dirs = viewdirs
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat, loss_entropy = batchify(fn, netchunk)(embedded, is_val, is_test)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + list(outputs_flat.shape[-2:])) # (B,N,K,4)
    
    return outputs, loss_entropy


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
    
    visualize_distribution = False
    if visualize_distribution:
        print(rays_o.shape) # (H,W,3)
        print(rays_d.shape) # (H,W,3)
        # check random surface point
        rays_o = rays_o[H//2,W//2,:]
        rays_d = rays_d[H//2,W//2,:]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)
    
    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        if k != 'loss_entropy' and  k != 'loss_entropy_uniformsample':
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    variances = []
    disps = []
    pts = []
    dists = []
    raw = []
    z_vals = []
    alpha_com = []
    alpha_mean = []
    rgb_mean = []
    alpha_var = []
    rgb_var = []
    weights_com = []
    alpha = []
    rgbss = []

    t = time.time()
    if render_poses.ndim == 3:
        for i, c2w in enumerate(tqdm(render_poses)):
            # print(i, time.time() - t)
            t = time.time()
            rgb, disp, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

            # negative to positive
            variance = torch.log(1 + torch.exp(var)) + 1e-05
            # variance = F.elu(var) + 1. 

            rgbs.append(rgb.cpu().numpy())
            variances.append(variance.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if i==0:
                print(rgb.shape, disp.shape)

            """
            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
            """

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    
    else:
        c2w = render_poses[:3,:4]
        rgb, disp, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        pts.append(extras['pts'].cpu().numpy())
        alpha.append(extras['alpha'].cpu().numpy())
        rgb_mean.append(extras['rgb_mean'].cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    pts = np.stack(pts, 0)
    alpha = np.stack(alpha, 0)
    rgb_mean = np.stack(rgb_mean, 0)

    return rgbs, disps, pts, alpha, rgb_mean


def render_path_train(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    variances = []
    disps = []
    pts = []
    
    raw = []
    z_vals = []
    alpha_com = []
    alpha_mean = []
    rgb_mean = []
    alpha_var = []
    rgb_var = []
    weights_com = []
    alpha = []
    rgb_mean_map = []
    rgbss = []

    t = time.time()
    if render_poses.ndim == 3:
        for i, c2w in enumerate(tqdm(render_poses)):
            # print(i, time.time() - t)
            t = time.time()
            rgb, disp, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

            # negative to positive
            variance = torch.log(1 + torch.exp(var)) + 1e-05
            # variance = F.elu(var) + 1. 

            rgbs.append(rgb.cpu().numpy())
            variances.append(variance.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if i==0:
                print(rgb.shape, disp.shape)

            """
            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
            """

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    
    else:
        c2w = render_poses[:3,:4]
        rgb, disp, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    args.embed_fn, args.input_ch = get_embedder(args.multires, args.i_embed)

    args.input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        args.embeddirs_fn, args.input_ch_views = get_embedder(args.multires_views, args.i_embed)
    args.output_ch = 5 if args.N_importance > 0 else 4
    args.skips = [args.netdepth / 2] 
    args.device = device
    model = NeRF_Flows(args)
    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, is_val, is_test : run_network(inputs, viewdirs, network_fn, is_val, is_test,
                                                                embed_fn=args.embed_fn,
                                                                embeddirs_fn=args.embeddirs_fn,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.dataname, args.type_flows, args.expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        if args.index_step == -1:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, '{:06d}_{:02d}.tar'.format(args.index_step, 1))
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # optimizer_unc.load_state_dict(ckpt['optimizer_unc_state_dict'])
        # Load model
        pretrained_dict = ckpt['network_fn_state_dict']
        # for k,v in pretrained_dict.items():
        #     print('loaded weights at ', k)
        model_dict = model.state_dict()
        # for k,v in model_dict.items():
        #     print('net weights at ', k)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        

    else:
        print('No reloading')

    ##########################

    render_kwargs_train = {
        'is_train' : args.is_train,
        'uniformsample' : args.uniformsample,
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'K_samples' : args.K_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['is_train'] = False
    render_kwargs_test['uniformsample'] = False
    render_kwargs_test['retraw'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, num_random_samples for each point, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e1]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, K, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            # np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3], dists[...,None])  # [N_rays, N_samples, K]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, alpha.shape[-1])), 1.-alpha + 1e-10], -2), -2)[:, :-1, :] # [N_rays, N_samples, K]
    rgb_map = torch.sum(weights[...,None] * rgb, -3)  # [N_rays, K, 3]
    rgb_map = rgb_map.transpose(-1,-2) # [N_rays, 3, K]

    depth_map = torch.sum(weights * z_vals[...,None], -2) # [N_rays, K, 3]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map) + 1e-10, depth_map / (torch.sum(weights, -2) + 1e-10) + 1e-10)
    acc_map = torch.sum(weights, -2)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[:,None,:])

    return rgb_map, disp_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                is_train,
                uniformsample,
                retraw=False,
                lindisp=False,
                K_samples=0,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    ## original 
    # t_vals = torch.linspace(0., 1., steps=N_samples)
    # option 2 for LF dataset
    t_vals = torch.cat([torch.linspace(0., 0.5, steps=97)[:-1],torch.linspace(0.5, 1., steps=32)],0)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            # np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts0 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    ## process
    is_test = not is_train
    raw, loss_entropy = network_query_fn(pts0, viewdirs, network_fn, is_val= False, is_test=is_test) # alpha_mean.shape (B,N,1)

    rgbs_map, disp_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    
    ret = {'rgb_map' : rgbs_map, 'disp_map' : disp_map, 'depth_map' : depth_map}
    
    if is_train:
        ret['raw'] = raw
        ret['loss_entropy'] = loss_entropy
        ret['pts'] = pts0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--dataname", type=str, default='leaves',
                        help='data name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options optimize_skip
    parser.add_argument("--is_train", action='store_true', 
                        help='train or evaluate')
    parser.add_argument("--uniformsample", action='store_true', 
                        help='use uniformsample points to train or not')
    parser.add_argument("--optimize_global", action='store_true', 
                        help='optimize_global or not')
    parser.add_argument("--optimize_skip", type=int, default=2, 
                        help='optimize_skip or not')
    parser.add_argument("--use_prior", action='store_true', 
                        help='use_prior or not')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    

    parser.add_argument("--model", type=str, default=None, 
                        help='model name')
    parser.add_argument("--N_rand", type=int, default=512, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_unc", type=float, default=5e-4,
                        help='learning rate') 
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    
    # flow options
    parser.add_argument("--type_flows", type=str, default='no_flow', choices=['planar', 'IAF', 'realnvp', 'glow', 'orthogonal', 'householder',
                                                                          'triangular', 'no_flow'],
                        help="""Type of flows to use, no flows can also be selected""")
    parser.add_argument("--n_flows", type=int, default=4, 
                        help='num of flows in normalizing flows')
    parser.add_argument("--n_hidden", type=int, default=128, 
                        help='channels per layer in normalizing flow network')
    parser.add_argument("--h_alpha_size", type=int, default=32, 
                        help='dims of h_context to normalizing flow network')
    parser.add_argument("--h_rgb_size", type=int, default=64, 
                        help='dims of h_context to normalizing flow network')
    parser.add_argument("--z_size", type=int, default=4, 
                        help='dims of samples from base distribution of normalizing flow network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--K_samples", type=int, default=64, 
                        help='number of monto-carlo samples per points')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--beta1", type=float, default=0.,
                        help='beta for balancing entropy loss and nll loss')
    parser.add_argument("--beta_u", type=float, default=0.1,
                        help='beta_uniformsample for balancing entropy loss and nll loss')
    parser.add_argument("--beta_p", type=float, default=0.05,
                        help='beta_prior for balancing entropy loss and nll loss')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    
    parser.add_argument("--colmap_depth", action='store_true', 
                        help='fraction of img taken for central crops') 
    parser.add_argument("--depth_lambda", type=float, default=0.1, 
                        help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000000, 
                        help='frequency of render_poses video saving')

    # emsemble setting
    parser.add_argument("--index_ensembles",   type=int, default=1, 
                        help='num of networks in ensembles')
    parser.add_argument("--index_step",   type=int, default=-1, 
                        help='step of weights to load in ensembles')


    return parser


def train(args):

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")
    print('building exp:',args.expname)

    # Load data
    if args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        
        if args.dataname == 'basket':
            # 4 views
            i_train = list(np.arange(43,50,2))
            i_val = list(np.arange(44,50,2))
            i_val_internal = list(np.arange(44,50,2))
        
        elif args.dataname == 'africa':
            # 5 views
            i_train = list(np.arange(5,14,2))
            i_val_internal = list(np.arange(6,14,2))
            i_val = list(np.arange(6,14,2))
        
        elif args.dataname == 'statue':
            # 5 views
            i_train = list(np.arange(67,76,2))
            i_val_internal = list(np.arange(68,76,2))
            i_val = list(np.arange(68,76,2))
        
        elif args.dataname == 'torch':
            # 5 views
            i_train = list(np.arange(8,17,2))
            i_val = list(np.arange(9,17,2))
            i_val_internal = list(np.arange(9,17,2))


        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    os.makedirs(os.path.join(args.basedir, args.dataname, args.type_flows, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    ##### Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand 
    N_depth = 128
    # for synthetic scenes with large area white background, if the network overfits to this background at the beginning, it makes the result very bad.
    # so authers used central croped object images for first ~1000 iters, which require us to sample from a single image at a time
    # or you can just increase batch size or trying other optimizers such as radam or ranger might help.
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # poses = poses[i_train,:,:]
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # train
        rays_rgb_train = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb_train = np.reshape(rays_rgb_train, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_train = rays_rgb_train.astype(np.float)
        print('rays_rgb_train.shape:', rays_rgb_train.shape)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_train)

        # val
        rays_rgb_val = np.stack([rays_rgb[i] for i in i_val_internal], 0) # val images only
        rays_rgb_val = np.reshape(rays_rgb_val, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_val = rays_rgb_val.astype(np.float)
        print('shuffle rays')
        np.random.shuffle(rays_rgb_val)

        print('done')
        i_batch_train = 0
        i_batch_val = 0

        rays_depth = None
        if args.colmap_depth:
            print('get depth rays')
            rays_depth_list = []
            for i in i_train:
                rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i,:3,:4], depth_gts[i]['coord']), axis=0) # 2 x N x 3
                # print(rays_depth.shape)
                rays_depth = np.transpose(rays_depth, [1,0,2])
                depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
                weights = np.repeat(depth_gts[i]['weight'][:,None,None], 3, axis=2) # N x 1 x 3
                rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1) # N x 4 x 3
                rays_depth_list.append(rays_depth)

            rays_depth = np.concatenate(rays_depth_list, axis=0)
            print('rays_weights mean:', np.mean(rays_depth[:,3,0]))
            print('rays_weights std:', np.std(rays_depth[:,3,0]))
            print('rays_weights max:', np.max(rays_depth[:,3,0]))
            print('rays_weights min:', np.min(rays_depth[:,3,0]))
            print('rays_depth.shape:', rays_depth.shape)
            rays_depth = rays_depth.astype(np.float)
            print('shuffle depth rays')
            np.random.shuffle(rays_depth)
            i_batch_depth = 0

            max_depth = np.max(rays_depth[:,3,0])
            rays_depth = torch.Tensor(rays_depth).to(device)

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb_train = torch.Tensor(rays_rgb_train).to(device)
        rays_rgb_val = torch.Tensor(rays_rgb_val).to(device)
    
    model = render_kwargs_train['network_fn']

    N_iters = 100000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(args.basedir, args.dataname, 'summaries', args.expname))

    start = start + 1
    idx = 0
    epoch = 0
    for i in trange(start, N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()

        ## load data
        # 1. Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb_train[i_batch_train:i_batch_train+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays_train, target_s = batch[:2], batch[2]

            i_batch_train += N_rand
            if i_batch_train >= rays_rgb_train.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb_train.shape[0])
                rays_rgb_train = rays_rgb_train[rand_idx]
                i_batch_train = 0
            
            # Random over all images on val
            batch = rays_rgb_val[i_batch_val:i_batch_val+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays_val, target_s_val = batch[:2], batch[2]

            i_batch_val += N_rand
            if i_batch_val >= rays_rgb_val.shape[0]:
                # print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb_val.shape[0])
                rays_rgb_val = rays_rgb_val[rand_idx]
                i_batch_val = 0
                
            if args.colmap_depth:
                batch_depth = rays_depth[i_batch_depth:i_batch_depth+N_depth]
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth = batch_depth[:2] # 2 x B x 3
                target_depth = batch_depth[2,:,0] # B
                ray_weights = batch_depth[3,:,0]
            
                i_batch_depth += N_depth
                if i_batch_depth >= rays_depth.shape[0]:
                    # print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_depth.shape[0])
                    rays_depth = rays_depth[rand_idx]
                    i_batch_depth = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays_train = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        if args.colmap_depth:
            N_batch = batch_rays_train.shape[1]
            batch_rays_train = torch.cat([batch_rays_train, batch_rays_depth], 1) # (2, 2 * N_rand, 3)

        #####  Core optimization loop1  #####
        rgbs, disp, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays_train,
                                                verbose=i < 10, retraw=False,
                                                **render_kwargs_train)

        ########## borrow from dsnerf 
        if args.colmap_depth:
            depth = torch.mean(depth,-1) # (N_rays, 3)
            rgbs = rgbs[:N_batch, :]
            disp = disp[:N_batch]
            depth, depth_col = depth[:N_batch], depth[N_batch:]
            extras = {x:extras[x][:N_batch] for x in extras}
        
        # compute mean and variance
        rgb_mean = torch.mean(rgbs,-1) # (N_rays, 3)
        mse_train = img2mse(rgb_mean, target_s)
        psnr_train = mse2psnr(mse_train)
            
        ############################# Multivariate kernel density estimation
        eps = 1e-05
        n = args.K_samples
        rgb_std = torch.std(rgbs, -1) * n / (n-1) # (N_rays, 3)

        H_sqrt = rgb_std.detach() * torch.pow(0.8/n,torch.tensor(-1/7)) + eps # (N_rays, 3)
        H_sqrt = H_sqrt[...,None] # (N_rays, 3, 1)
        r_P_C_1 = torch.exp( -((rgbs - target_s[...,None])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, 3, k]
        r_P_C_2 = torch.pow(torch.tensor(2*math.pi),-1.5) / H_sqrt # [N_rays, 3, 1]
        r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
        r_P_C_mean = r_P_C.mean(-1) + eps
        loss_nll = - torch.log(r_P_C_mean).mean()

        # loss_entropy
        loss_entropy = extras['loss_entropy'].mean()

        if args.beta1:
            loss = loss_nll + args.beta1 * loss_entropy
        else:
            loss = loss_nll

        if args.colmap_depth:
            depth_loss = img2mse(depth_col, target_depth)
            loss = loss + args.depth_lambda * depth_loss
            scalars_to_log['train/depth_loss'] = depth_loss.item()

        scalars_to_log['train/loss_entropy'] = loss_entropy.item()
        scalars_to_log['train/loss_nll'] = loss_nll.item()
        scalars_to_log['train/logprob'] = loss_nll.item()
        scalars_to_log['train/mse'] = mse_train.item()
        scalars_to_log['train/pnsr'] = psnr_train.item()
        scalars_to_log['train/loss'] = loss.item()

        time1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del rgbs, disp, extras
 
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        ################################

        dt = time.time()-time0
        scalars_to_log['iter_time'] = dt

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, '{:06d}_{:02d}.tar'.format(i, args.index_ensembles))
            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # tensorboard
        if i % args.i_img == 0 and i > start +1:
            # for train data
            idx_t = idx % len(i_train)
            idx_train = i_train[idx_t] 
            with torch.no_grad():
                rgbs, disps = render_path_train(torch.Tensor(poses[idx_train]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[idx_train]) # rgbs, (N, H, W, 3, k3)

            rgbs = rgbs.squeeze()
            disps = disps.squeeze()

            rgbs_mean = np.mean(rgbs,-1).reshape([H,W,3])
            disps_mean = np.mean(disps,-1).reshape([H,W,1])

            mse_ = (rgbs_mean - images[idx_train].cpu().numpy())**2
            heatmap_mse_ = cv2.applyColorMap(to8b(mse_), cv2.COLORMAP_JET)
            heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            n = rgbs.shape[-1]
            rgbs_std = np.std(rgbs, -1) * n / (n-1) # (H,W,3)
            heatmap_v = cv2.applyColorMap(to8b(rgbs_std), cv2.COLORMAP_JET)
            heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            img_pred = to8b(rgbs_mean.transpose(2,0,1))

            disps_mean = disps_mean / np.percentile(disps_mean,90)
            heatmap_disps = cv2.applyColorMap(to8b(disps_mean.reshape([H,W,1])), cv2.COLORMAP_MAGMA)
            img_disp_pred = cv2.cvtColor(heatmap_disps, cv2.COLOR_BGR2RGB).transpose(2,0,1)
            # img_disp_pred = to8b(disps_mean / np.percentile(disps_mean,80)).transpose(2,0,1)
            img_gt = to8b(images[idx_train].detach().cpu().numpy()).transpose(2,0,1)

            prefix='train/'
            writer.add_image(prefix + 'rgb_gt', img_gt, i)
            writer.add_image(prefix + 'rgb_pred', img_pred, i)
            writer.add_image(prefix + 'rgb_disp_pred', img_disp_pred, i)
            writer.add_image(prefix + 'heatmap_mse_', heatmap_mse_, i)
            writer.add_image(prefix + 'heatmap_v', heatmap_v, i)

            del rgbs, disps

            # for val data
            idx_v = idx % len(i_val)
            idx_val = i_val[idx_v]
            with torch.no_grad():
                rgbs, disps = render_path_train(torch.Tensor(poses[idx_val]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[idx_val]) # rgbs, (N, H*W, 3)

            rgbs = rgbs.squeeze()
            disps = disps.squeeze()

            rgbs_mean = np.mean(rgbs,-1).reshape([H,W,3])
            disps_mean = np.mean(disps,-1).reshape([H,W,1])

            mse_ = (rgbs_mean - images[idx_val].cpu().numpy())**2
            heatmap_mse_ = cv2.applyColorMap(to8b(mse_), cv2.COLORMAP_JET)
            heatmap_mse_ = cv2.cvtColor(heatmap_mse_, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            n = rgbs.shape[-1]
            rgbs_std = np.std(rgbs, -1) * n / (n-1) # (H,W,3)
            heatmap_v = cv2.applyColorMap(to8b(rgbs_std), cv2.COLORMAP_JET)
            heatmap_v = cv2.cvtColor(heatmap_v, cv2.COLOR_BGR2RGB).transpose(2,0,1)

            img_pred = to8b(rgbs_mean.transpose(2,0,1))

            disps_mean = disps_mean / np.percentile(disps_mean,90)
            heatmap_disps = cv2.applyColorMap(to8b(disps_mean.reshape([H,W,1])), cv2.COLORMAP_MAGMA)
            img_disp_pred = cv2.cvtColor(heatmap_disps, cv2.COLOR_BGR2RGB).transpose(2,0,1)
            # img_disp_pred = to8b(disps_mean / np.percentile(disps_mean,80)).transpose(2,0,1)
            img_gt = to8b(images[idx_val].detach().cpu().numpy()).transpose(2,0,1)

            prefix='val/'
            writer.add_image(prefix + 'rgb_gt', img_gt, i)
            writer.add_image(prefix + 'rgb_pred', img_pred, i)
            writer.add_image(prefix + 'rgb_disp_pred', img_disp_pred, i)
            writer.add_image(prefix + 'heatmap_mse_', heatmap_mse_, i)
            writer.add_image(prefix + 'heatmap_v', heatmap_v, i)

            idx += 1
        
        if i%args.i_print==0:
            if args.colmap_depth:
               tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} entropy: {loss_entropy.item()} depth: {depth_loss.item()} nll: {loss_nll.item()} PSNR: {psnr_train.item()}")
            else:
               tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} nll: {loss_nll.item()} PSNR: {psnr_train.item()}")
            # tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} kl_loss_alpha: {kl_loss_alpha.item()}  MSE: {mse_train.item()} PSNR: {psnr_train.item()}")
            for k in scalars_to_log:
                writer.add_scalar(k, scalars_to_log[k], i)

        global_step += 1

def test(args):

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        
        if args.dataname == 'basket':
            # 4 views
            i_train = list(np.arange(43,50,2))
            i_val = list(np.arange(44,50,2))
            i_val_internal = list(np.arange(44,50,2))
        
        elif args.dataname == 'africa':
            # 5 views
            i_train = list(np.arange(5,14,2))
            i_val = list(np.arange(6,14,2))
            i_val_internal = list(np.arange(6,14,2))
        
        elif args.dataname == 'statue':
            # 5 views
            i_train = list(np.arange(67,76,2))
            i_val_internal = list(np.arange(68,76,2))
            i_val = list(np.arange(68,76,2))
        
        elif args.dataname == 'torch':
            # 5 views
            i_train = list(np.arange(8,17,2))
            i_val = list(np.arange(9,17,2))
            i_val_internal = list(np.arange(9,17,2))

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    # Cast intrinsics to right types, change H,W
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    N_iters = 100000 + 1
    print('Begin')
    

    # Summary writers

    step = start + 1

    depth_mse_all = []
    depth_std_all = []
    depth_mae_all = []
    mse_all = []
    std_all = []
    mae_all = []
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    logprob_sum = 0
    depth_logprob_sum = 0
    delta_1_sum = 0  
    delta_2_sum = 0  
    delta_3_sum = 0  

    start = start + 1
    idx = 0

    model = render_kwargs_train['network_fn']
    model_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_param_num:',model_param_num)

    ############# save results ##############
    if 1:
        i_test = i_val_internal

        for i, img_i in enumerate(tqdm(i_test)):

            # Random from one image
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            print('img:',img_i)
            print('k_samples:',args.K_samples)

            t0 = time.time() 

            #####  Core optimization loop  #####
            with torch.no_grad():
                rgbs, disps = render_path_train(pose, hwf, args.chunk, render_kwargs_test) # rgbs, (N, H, W, 3, k3)
            
            eps_time = time.time() - t0
            # eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            print('inference time:',eps_time)
            
            # rgb maybe > 1, which arise due to val training, so 
            # rgbs = np.minimum(rgbs, 1.0)

            rgbs = rgbs.squeeze()
            disps = disps.squeeze()
    
            testsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'testset_{:06d}'.format(step))
            os.makedirs(testsavedir, exist_ok=True)
            target_s = images[img_i].cpu().numpy()

            # # save all disp samples as video
            # moviesavedir = testsavedir + '/video_depth'
            # os.makedirs(moviesavedir, exist_ok=True)
            # for t in range(disps.shape[-1]):
            #     disp = cv2.cvtColor(disps[...,t], cv2.COLOR_BGR2RGB)
            #     cv2.imwrite(moviesavedir+'/{:02d}_depth_{:02d}.png'.format(img_i,t), to8b(disp))
            # imageio.mimwrite(moviesavedir + '/depth_{:02d}.mp4'.format(img_i), to8b(disps.transpose(2,0,1)), fps=8, quality=8)
            # exit(1)

            # # save all rgb samples as video
            # moviesavedir = testsavedir + '/video'
            # os.makedirs(moviesavedir, exist_ok=True)
            # for t in range(rgbs.shape[-1]):
            #     rgb = cv2.cvtColor(rgbs[...,t], cv2.COLOR_BGR2RGB)
            #     cv2.imwrite(moviesavedir+'/{:02d}_pred_{:02d}.png'.format(img_i,t), to8b(rgb))
            # imageio.mimwrite(moviesavedir + '/rgb_{:02d}.mp4'.format(img_i), to8b(rgbs.transpose(3,0,1,2)), fps=8, quality=8)

            #### load depth gt
            depth = 0
            if depth:
                rootsavedir = os.path.join(args.basedir, args.dataname)
                os.makedirs(rootsavedir, exist_ok=True)
                with open(rootsavedir + '/depth_gt_{:02d}.npy'.format(img_i),'rb') as f:
                    disps_gt = np.load(f)
            
                # normalize disp to 0-1
                disps_gt = (disps_gt - disps_gt.min()) / (disps_gt.max() - disps_gt.min()) 
            disps = (disps - disps.min()) / (disps.max() - disps.min()) 

            # cut edge 
            if args.dataname == 'basket':
                rgbs = rgbs[20:-20,20:-20,...]
                disps = disps[20:-20,20:-20,...]
                # disps_gt = disps_gt[20:-20,20:-20]
                target_s = target_s[20:-20,20:-20,...]
                H, W = rgbs.shape[:2]
            
            if depth:
                ##############################  depth map  ###############################
                # measure uncertainty for depth
                ##########################################################################
                ### 1. depth error
                disps_mean = np.mean(disps,-1) # (H,W)
                depth_mse = (disps_mean-disps_gt)**2
                depth_mae = np.abs(disps_mean-disps_gt)
                # delta 
                delta_thr1 = np.power(1.25,1)
                delta_thr2 = np.power(1.25,2)
                delta_thr3 = np.power(1.25,3)
                a = np.maximum(disps_mean.reshape(-1) / disps_gt.reshape(-1), disps_gt.reshape(-1) / disps_mean.reshape(-1))
                b1 = a[a < delta_thr1]
                b2 = a[a < delta_thr2]
                b3 = a[a < delta_thr3]
                print('delta 1:',len(b1) / len(a))
                print('delta 2:',len(b2) / len(a))
                print('delta 3:',len(b3) / len(a))
                print('depth rmse:',np.sqrt(depth_mse.mean()))
                print('depth mae:',depth_mae.mean())
                ### 2. depth uncertainty
                # 2.1 nll
                eps = 1e-05
                n = disps.shape[-1]
                disps_std = np.std(disps[...,:-1], -1) * n / (n-1) # (N_rays, 3)
                H_sqrt = disps_std * np.power(0.8/n,-1/7) + eps # (N_rays, 3)
                H_sqrt = H_sqrt[...,None] # (N_rays, 3, 1)
                r_P_C_1 = np.exp( -((disps - disps_gt[...,None])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, 3, k]
                r_P_C_2 = np.power(2*math.pi,-1.5) / H_sqrt # [N_rays, 3, 1]
                r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
                r_P_C_mean = r_P_C.mean(-1) + eps
                logprob = np.log(r_P_C_mean).mean()
                print('depth logprob:',logprob)
                depth_logprob_sum += logprob

                # 2.2 ause
                ause_depth_dir = testsavedir + '/depth_ause'
                os.makedirs(ause_depth_dir, exist_ok=True)
                # rmse 
                depth_mse_r = depth_mse.reshape(-1) # (N,)
                depth_std_r = disps_std.reshape(-1) # (N,)
                ratio_removed = np.linspace(0, 1, 100, endpoint=False)
                rmse_by_err, rmse_by_std = sparsification_plot(torch.tensor(depth_std_r), torch.tensor(depth_mse_r), uncert_type='v', err_type='rmse')
                ause = np.trapz(rmse_by_std - rmse_by_err, ratio_removed)
                ausc = np.trapz(rmse_by_std, ratio_removed)
                with open(ause_depth_dir + '/rmse_by_error_{:02d}.npy'.format(img_i),'wb') as f:
                    np.save(f,rmse_by_err)
                with open(ause_depth_dir + '/rmse_by_std_{:02d}.npy'.format(img_i),'wb') as f:
                    np.save(f,rmse_by_std)
                print('- depth AUSC metric std - rmse is {:.5f}.'.format(ausc))
                print('- depth AUSE metric std - rmse is {:.5f}.'.format(ause))
                plt.clf()
                plt.plot(ratio_removed, rmse_by_err, '--')
                plt.plot(ratio_removed, rmse_by_std, '-r')
                plt.grid()
                plt.savefig(ause_depth_dir+'/rmse_ause_{:02d}.png'.format(img_i))
                # mae
                depth_mae_r = depth_mae.reshape(-1) # (N,)
                depth_std_r = disps_std.reshape(-1) # (N,)
                ratio_removed = np.linspace(0, 1, 100, endpoint=False)
                mae_by_err, mae_by_std = sparsification_plot(torch.tensor(depth_std_r), torch.tensor(depth_mae_r), uncert_type='v', err_type='mae')
                ause = np.trapz(mae_by_std - mae_by_err, ratio_removed)
                ausc = np.trapz(mae_by_std, ratio_removed)
                with open(ause_depth_dir + '/mae_by_error_{:02d}.npy'.format(img_i),'wb') as f:
                    np.save(f,mae_by_err)
                with open(ause_depth_dir + '/mae_by_std_{:02d}.npy'.format(img_i),'wb') as f:
                    np.save(f,mae_by_std)
                print('- depth AUSC metric std - mae is {:.5f}.'.format(ausc))
                print('- depth AUSE metric std - mae is {:.5f}.'.format(ause))
                plt.clf()
                plt.plot(ratio_removed, mae_by_err, '--')
                plt.plot(ratio_removed, mae_by_std, '-r')
                plt.grid()
                plt.savefig(ause_depth_dir+'/mae_ause{:02d}.png'.format(img_i))

                delta_1_sum += len(b1) / len(a)
                delta_2_sum += len(b2) / len(a)
                delta_3_sum += len(b3) / len(a)
                depth_mse_all.append(depth_mse_r)
                depth_mae_all.append(depth_mae_r)
                depth_std_all.append(depth_std_r)

                depth_mae_map = to8b(depth_mae)
                heatmap_mae_ = cv2.applyColorMap(depth_mae_map, cv2.COLORMAP_MAGMA)
                cv2.imwrite(testsavedir+'/disp_{:02d}_mae.png'.format(img_i), heatmap_mae_)

            # disps_mean = np.mean(disps,-1)
            disps_mean = disps[...,-1]
            # disps_show = cv2.applyColorMap(to8b(disps_mean.reshape([H,W,1]) / np.percentile(disps_mean,90)), cv2.COLORMAP_MAGMA)
            # cv2.imwrite(testsavedir+'/disp_{:02d}.png'.format(img_i), disps_show) 
            disps_show = to8b(disps_mean.reshape([H,W,1]) / np.percentile(disps_mean,93))
            cv2.imwrite(testsavedir+'/disp_{:02d}.png'.format(img_i), disps_show)

            disps_show = to8b(disps_mean.reshape([H,W,1]) / np.percentile(disps_mean,95))
            cv2.imwrite(testsavedir+'/disp_{:02d}_2.png'.format(img_i), disps_show)

            disps_std = np.std(disps,-1)
            heatmap_depth_std = cv2.applyColorMap(to8b(disps_std / np.percentile(disps_std,99)), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/disp_{:02d}_std.png'.format(img_i), heatmap_depth_std)

            heatmap_depth_std = cv2.applyColorMap(to8b(disps_std / np.percentile(disps_std,99.9)), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/disp_{:02d}_std1.png'.format(img_i), heatmap_depth_std)

            heatmap_depth_std = cv2.applyColorMap(to8b(disps_std / np.percentile(disps_std,99.99)), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/disp_{:02d}_std2.png'.format(img_i), heatmap_depth_std)
            
            ##############################  rgb map  ###############################
            # measure uncertainty for rgb
            ##########################################################################
            ### 1. rgb error
            rgbs_mean = np.mean(rgbs,-1) # (B,H,W,3)
            rgb_mse = (rgbs_mean-target_s)**2
            rgb_mse_intepolate = (rgbs[...,-1]-target_s)**2
            rgb_mae = np.abs(rgbs_mean-target_s)
            ### 2. rgb uncertainty
            # 2.1 nll
            eps = 1e-05
            n = rgbs.shape[-1]
            rgb_std = np.std(rgbs, -1) * n / (n-1) # (N_rays, 3)
            H_sqrt = rgb_std * np.power(0.8/n,-1/7) + eps # (N_rays, 3)
            H_sqrt = H_sqrt[...,None] # (N_rays, 3, 1)
            r_P_C_1 = np.exp( -((rgbs - target_s[...,None])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, 3, k]
            r_P_C_2 = np.power(2*math.pi,-1.5) / H_sqrt # [N_rays, 3, 1]
            r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
            r_P_C_mean = r_P_C.mean(-1) + eps
            logprob = np.log(r_P_C_mean).mean()
            print('rgb logprob:',logprob)
            logprob_sum += logprob

            # 2.2 ause
            ause_rgb_dir = testsavedir + '/rgb_ause'
            os.makedirs(ause_rgb_dir, exist_ok=True)
            # rmse 
            mse_r = np.mean(rgb_mse,-1).reshape(-1) # (N,)
            std_r = np.mean(rgb_std,-1).reshape(-1) # (N,)
            ratio_removed = np.linspace(0, 1, 100, endpoint=False)
            rmse_by_err, rmse_by_std = sparsification_plot(torch.tensor(std_r), torch.tensor(mse_r), uncert_type='v', err_type='rmse')
            with open(ause_rgb_dir + '/rmse_by_error_{:02d}.npy'.format(img_i),'wb') as f:
                np.save(f,rmse_by_err)
            with open(ause_rgb_dir + '/rmse_by_std_{:02d}.npy'.format(img_i),'wb') as f:
                np.save(f,rmse_by_std)
            ause = np.trapz(rmse_by_std - rmse_by_err, ratio_removed)
            ausc = np.trapz(rmse_by_std, ratio_removed)
            print('- rgb AUSC metric std - rmse is {:.5f}.'.format(ausc))
            print('- rgb AUSE metric std - rmse is {:.5f}.'.format(ause))
            plt.clf()
            plt.plot(ratio_removed, rmse_by_err, '--')
            plt.plot(ratio_removed, rmse_by_std, '-r')
            plt.grid()
            plt.savefig(ause_rgb_dir+'/rmse_ause_{:02d}.png'.format(img_i))
            # mae
            mae_r = np.mean(rgb_mae,-1).reshape(-1) # (N,)
            std_r = np.mean(rgb_std,-1).reshape(-1) # (N,)
            ratio_removed = np.linspace(0, 1, 100, endpoint=False)
            mae_by_err, mae_by_std = sparsification_plot(torch.tensor(std_r), torch.tensor(mae_r), uncert_type='v', err_type='mae')
            with open(ause_rgb_dir + '/mae_by_error_{:02d}.npy'.format(img_i),'wb') as f:
                np.save(f,mae_by_err)
            with open(ause_rgb_dir + '/mae_by_std_{:02d}.npy'.format(img_i),'wb') as f:
                np.save(f,mae_by_std)
            ause = np.trapz(mae_by_std - mae_by_err, ratio_removed)
            ausc = np.trapz(mae_by_std, ratio_removed)
            print('- rgb AUSC metric std - mae is {:.5f}.'.format(ausc))
            print('- rgb AUSE metric std - mae is {:.5f}.'.format(ause))
            plt.clf()
            plt.plot(ratio_removed, mae_by_err, '--')
            plt.plot(ratio_removed, mae_by_std, '-r')
            plt.grid()
            plt.savefig(ause_rgb_dir+'/mae_ause{:02d}.png'.format(img_i))

            ##############################  image quality  ###############################
            # measure image quality for rgb
            ##############################################################################
            # psnr, ssim, lpips
            psnr = -10. * np.log(rgb_mse.mean()) / np.log(10.)
            # psnr = -10. * np.log(rgb_mse_intepolate.mean()) / np.log(10.)
            print('image_i:{:02d} psnr:{:.3f}'.format(img_i,psnr))
            psnr_sum += psnr
            ssim = SSIM(rgbs_mean, target_s, multichannel=True)
            # ssim = SSIM(rgbs[...,-1], target_s, multichannel=True)
            print('image_i:{:02d} ssim:{:.3f}'.format(img_i,ssim))
            ssim_sum += ssim
            import lpips
            loss_fn_alex = lpips.LPIPS(net='alex')
            lpips = loss_fn_alex(torch.Tensor(rgbs_mean.transpose(2,0,1)), torch.Tensor(target_s.transpose(2,0,1))).squeeze()
            # lpips = loss_fn_alex(torch.Tensor(rgbs[...,-1].transpose(2,0,1)), torch.Tensor(target_s.transpose(2,0,1))).squeeze()
            print('image_i:{:02d} lpips:{:.3f}'.format(img_i,lpips.detach().cpu().numpy()))
            lpips_sum += lpips

            ###############################  show  ###############################
            rgb_mean = cv2.cvtColor(rgbs_mean, cv2.COLOR_BGR2RGB)
            cv2.imwrite(testsavedir+'/{:02d}_pred.png'.format(img_i), to8b(rgb_mean))

            mse_map = to8b(rgb_mse*1)
            heatmap_mse_ = cv2.applyColorMap(mse_map, cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_mse.png'.format(img_i), heatmap_mse_)

            mae_map = to8b(rgb_mae*2)
            heatmap_mae_ = cv2.applyColorMap(mae_map, cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_mae.png'.format(img_i), heatmap_mae_)

            mae_map = to8b(rgb_mae*3)
            heatmap_mae_ = cv2.applyColorMap(mae_map, cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_mae2.png'.format(img_i), heatmap_mae_)

            heatmap_std = cv2.applyColorMap(to8b(rgb_std*5), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_std.png'.format(img_i), heatmap_std)

            heatmap_std = cv2.applyColorMap(to8b(rgb_std*10), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_std1.png'.format(img_i), heatmap_std)

            heatmap_std = cv2.applyColorMap(to8b(rgb_std*20), cv2.COLORMAP_MAGMA)
            cv2.imwrite(testsavedir+'/{:02d}_std2.png'.format(img_i), heatmap_std)

            mse_all.append(mse_r)
            mae_all.append(mae_r)
            std_all.append(std_r)

        if depth:
            # rmse
            mse_total = np.stack(depth_mse_all,0).reshape(-1)
            mae_total = np.stack(depth_mae_all,0).reshape(-1)
            std_total = np.stack(depth_std_all,0).reshape(-1)
            rmse_by_err, rmse_by_std = sparsification_plot(torch.tensor(std_total), torch.tensor(mse_total), uncert_type='v', err_type='rmse')
            ause = np.trapz(rmse_by_std - rmse_by_err, ratio_removed)
            ausc = np.trapz(rmse_by_std, ratio_removed)
            with open(ause_depth_dir + '/rmse_by_error_avg.npy','wb') as f:
                np.save(f,rmse_by_err)
            with open(ause_depth_dir + '/rmse_by_std_avg.npy','wb') as f:
                np.save(f,rmse_by_std)
            print('- depth AUSC metric std - rmse is {:.5f}.'.format(ausc))
            print('- depth AUSE metric std - rmse is {:.5f}.'.format(ause))
            plt.clf()
            plt.plot(ratio_removed, rmse_by_err, '--')
            plt.plot(ratio_removed, rmse_by_std, '-r')
            plt.grid()
            plt.savefig(ause_depth_dir+'/rmse_ause_all.png')
            # mae
            mae_by_err, mae_by_std = sparsification_plot(torch.tensor(std_total), torch.tensor(mae_total), uncert_type='v', err_type='mae')
            ause = np.trapz(mae_by_std - mae_by_err, ratio_removed)
            ausc = np.trapz(mae_by_std, ratio_removed)
            with open(ause_depth_dir + '/mae_by_error_avg.npy','wb') as f:
                np.save(f,mae_by_err)
            with open(ause_depth_dir + '/mae_by_std_avg.npy','wb') as f:
                np.save(f,mae_by_std)
            print('- depth AUSC metric std - mae is {:.5f}.'.format(ausc))
            print('- depth AUSE metric std - mae is {:.5f}.'.format(ause))
            plt.clf()
            plt.plot(ratio_removed, mae_by_err, '--')
            plt.plot(ratio_removed, mae_by_std, '-r')
            plt.grid()
            plt.savefig(ause_depth_dir+'/mae_ause_all.png'.format(img_i))

            print('avg depth logprob for i_test:', depth_logprob_sum / len(depth_mse_all))
            print('avg depth delta_1 for i_test:', delta_1_sum / len(depth_mse_all))
            print('avg depth delta_2 for i_test:', delta_2_sum / len(depth_mse_all))
            print('avg depth delta_3 for i_test:', delta_3_sum / len(depth_mse_all))
            print('avg depth RMSE for i_test:', np.sqrt(np.stack(depth_mse_all,0).mean()))
            print('avg depth MAE for i_test:', np.stack(depth_mae_all,0).mean())
        
        # rmse
        mse_total = np.stack(mse_all,0).reshape(-1)
        mae_total = np.stack(mae_all,0).reshape(-1)
        std_total = np.stack(std_all,0).reshape(-1)
        rmse_by_err, rmse_by_std = sparsification_plot(torch.tensor(std_total), torch.tensor(mse_total), uncert_type='v', err_type='rmse')
        ause = np.trapz(rmse_by_std - rmse_by_err, ratio_removed)
        ausc = np.trapz(rmse_by_std, ratio_removed)
        with open(ause_rgb_dir + '/rmse_by_error_avg.npy','wb') as f:
            np.save(f,rmse_by_err)
        with open(ause_rgb_dir + '/rmse_by_std_avg.npy','wb') as f:
            np.save(f,rmse_by_std)
        print('- AUSC metric std - rmse is {:.5f}.'.format(ausc))
        print('- AUSE metric std - rmse is {:.5f}.'.format(ause))
        plt.clf()
        plt.plot(ratio_removed, rmse_by_err, '--')
        plt.plot(ratio_removed, rmse_by_std, '-r')
        plt.grid()
        plt.savefig(ause_rgb_dir+'/rmse_ause_all.png')
        # mae
        mae_by_err, mae_by_std = sparsification_plot(torch.tensor(std_total), torch.tensor(mae_total), uncert_type='v', err_type='mae')
        ause = np.trapz(mae_by_std - mae_by_err, ratio_removed)
        ausc = np.trapz(mae_by_std, ratio_removed)
        with open(ause_rgb_dir + '/mae_by_error_avg.npy','wb') as f:
            np.save(f,mae_by_err)
        with open(ause_rgb_dir + '/mae_by_std_avg.npy','wb') as f:
            np.save(f,mae_by_std)
        print('- AUSC metric std - mae is {:.5f}.'.format(ausc))
        print('- AUSE metric std - mae is {:.5f}.'.format(ause))
        plt.clf()
        plt.plot(ratio_removed, mae_by_err, '--')
        plt.plot(ratio_removed, mae_by_std, '-r')
        plt.grid()
        plt.savefig(ause_rgb_dir+'/mae_ause_all.png'.format(img_i))

        print('avg psnr for mse_all:', psnr_sum / len(mse_all))
        print('avg ssim for mse_all:', ssim_sum / len(mse_all))
        print('avg lpips for mse_all:', lpips_sum / len(mse_all))
        print('avg logprob for i_test:', logprob_sum / len(mse_all))

        exit(1)
    
    ############# save results ##############
    if 0:
        print('i_val_internal:',i_val_internal) 
        print('i_val_external:',i_val_external)
        i_test = i_val_external
        for i, img_i in enumerate(tqdm(i_test)):

            # Random from one image
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            t0 = time.time()

            testsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'testset_{:06d}'.format(args.index_step), 'videos')
            os.makedirs(testsavedir, exist_ok=True)

            #####  Core optimization loop  #####
            with torch.no_grad():
                rgbs, disps = render_path_train(pose, hwf, args.chunk, render_kwargs_test) # rgbs, (N, H, W, 3, k3)
            
            rgbs = rgbs.squeeze()
            disps = disps.squeeze()
            rgbs = rgbs.transpose(3,0,1,2)
            disps = disps.transpose(2,0,1)

            for t in range(rgbs.shape[0]):
                rgb = cv2.cvtColor(rgbs[t,...], cv2.COLOR_BGR2RGB)
                cv2.imwrite(testsavedir+'/{:02d}_pred_{:02d}.png'.format(img_i,t), to8b(rgb))

            rgbs = np.concatenate([rgbs,rgbs[:-1,...][::-1,...]],0)
            disps = np.concatenate([disps,disps[:-1,...][::-1,...]],0)
            print(rgbs.shape)
    
            imageio.mimwrite(testsavedir + '/uncertainty_rgb_{:02d}.mp4'.format(img_i), to8b(rgbs), fps=8, quality=8)
            imageio.mimwrite(testsavedir + '/uncertainty_disp_{:02d}.mp4'.format(img_i), to8b(disps / np.max(disps)), fps=8, quality=8)
        
        exit(1)

    # ############# convert to mesh using marching cubes ################
    if 0:   
        #################### load model 
        # embedding_xyz = Embedding(3, 10)
        # embedding_dir = Embedding(3, 4)

        embedding_xyz, _ = get_embedder(args.multires, args.i_embed)
        embedding_dir, _ = get_embedder(args.multires_views, args.i_embed)

        model = render_kwargs_train['network_fn']
        model.eval()

        # print(model.module.alpha_feature)

        # ckpt_path = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, '{:06d}_{:02d}.tar'.format(args.index_step, 1))
        # print('Reloading from', ckpt_path)
        # ckpt = torch.load(ckpt_path)

        # start = ckpt['global_step']
        # pretrained_dict = ckpt['network_fn_state_dict']
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict) 
        # model.load_state_dict(model_dict)

        #################### raw volume ####################
        ### Tune these parameters until the whole object lies tightly in range with little noise ###
        N = 256 # controls the resolution, set this number small here because we're only finding
                # good ranges here, not yet for mesh reconstruction; we can set this number high
                # when it comes to final reconstruction.
        xmin, xmax = -1.2, 1.2 # left/right range
        ymin, ymax = -1.2, 1.2 # forward/backward range
        zmin, zmax = -1.2, 1.2 # up/down range
        ## Attention! the ranges MUST have the same length!
        threshold = 0.3 # controls the noise (lower=maybe more noise; higher=some mesh might be missing)

        x = np.linspace(xmin, xmax, N)
        y = np.linspace(ymin, ymax, N)
        z = np.linspace(zmin, zmax, N)

        xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
        dir_ = torch.zeros_like(xyz_).cuda()
        # dir_ = rays_rgb_train.mean(0)
        # dir_ = dir_[1]
        # dir_ = dir_[None,...].expand_as(xyz_)
        # batch = rays_rgb_train[:xyz_.shape[0]] # [B, 2+1, 3*?]
        # batch = torch.transpose(batch, 0, 1)
        # rays_o, rays_d = batch[:2]
        # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # rays_d = torch.reshape(rays_d, [-1,3]).float()
        # # ndc, for forward facing scenes
        # _, dir_ = ndc_rays(H, W, focal, 1., rays_o, rays_d)

        with torch.no_grad():
            B = xyz_.shape[0]
            out_chunks = []
            chunk = 1024*8
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # (N, embed_xyz_channels)
                dir_embedded = embedding_dir(dir_[i:i+chunk]) # (N, embed_dir_channels)
                xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
                out_chunks += [model.module.sample(xyzdir_embedded)]
            rgbsigma = torch.cat(out_chunks, 0)  #  (NNN, K, 4)
            print('output size',rgbsigma.shape)
        
        # sigma_mean = rgbsigma[:, -2].cpu().numpy() # (N*N*N)
        # sigma_std = F.softplus(rgbsigma[...,-1]).cpu().numpy() + 1e-05 # (N*N*N)
        # k = 32
        # eps = np.random.randn(k)
        # sigma_k = sigma_mean[...,None] + sigma_std[...,None] * eps # (N*N*N, k)
        sigma_k = rgbsigma[..., -1].cpu().numpy() #  (NNN, K)
        k = sigma_k.shape[-1]
        sigma_k = np.maximum(sigma_k, 0) # (N*N*N, k)
        sigma_k = sigma_k.reshape(N, N, N, k)

        print('fraction occupied', np.mean(sigma_k[...,0] > threshold))

        # The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
        def pyrender_pose(mesh, k):
            ############# Save out video with pyrender
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender
            from load_blender import pose_spherical

            scene = pyrender.Scene()
            # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

            # camera_pose = pose_spherical(-20., -40., 1.).cpu().numpy()
            camera_pose1 = pose_spherical(-110, -40., 1.).cpu().numpy()
            camera_pose2 = pose_spherical(-20, -40., 1.).cpu().numpy()
            camera_pose3 = pose_spherical(70, -40., 1.).cpu().numpy()
            camera_pose4 = pose_spherical(160, -40., 1.).cpu().numpy()
            camera_pose = [camera_pose1,camera_pose2,camera_pose3,camera_pose4]

            ## Set up the light -- a point light in the same spot as the camera
            light2 = pyrender.PointLight(color=np.ones(3), intensity=4.0)
            nl = pyrender.Node(light=light2, matrix=camera_pose2)
            scene.add_node(nl)
            light4 = pyrender.PointLight(color=np.ones(3), intensity=4.0)
            nl = pyrender.Node(light=light4, matrix=camera_pose4)
            scene.add_node(nl)

            for i in range(len(camera_pose)):
                # Render the scene
                print('camera:',i)
                nc = pyrender.Node(camera=camera, matrix=camera_pose[i])
                scene.add_node(nc)
                r = pyrender.OffscreenRenderer(640, 480)
                color, depth = r.render(scene)
                f_color = meshsavedir + '/lego_mesh_cam{:02d}_k{:02d}_t{:.3f}.png'.format(i, k, threshold)
                imageio.imwrite(f_color, color)
                r.delete()
                scene.remove_node(nc)
        
        import mcubes, trimesh
        for m in range(k):
            vertices, triangles = mcubes.marching_cubes(sigma_k[...,m], threshold)
            print('done', vertices.shape, triangles.shape)
            meshsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'testset_{:06d}'.format(args.index_step), 'interpolation')
            os.makedirs(meshsavedir, exist_ok=True)
            filename = meshsavedir + '/k{:02d}_t{:.3f}_mesh.dae'.format(m, threshold)
            mcubes.export_mesh(vertices, triangles, filename)
            mesh = trimesh.Trimesh(vertices/N - 0.5, triangles)
            
            # mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
            # mesh.export(filename)
            # exit(1)
            # mesh.show()
            pyrender_pose(mesh, m)
            print('finish no:',m)

        for i in range(4):
            imgs = []
            meshsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'testset_{:06d}'.format(args.index_step), 'interpolation')
            for m in range(k):
                f_color = meshsavedir + '/lego_mesh_cam{:02d}_k{:02d}_t{:.3f}.png'.format(i, m, threshold)
                imgs.append(imageio.imread(f_color))
            f_gif = meshsavedir + '/lego_mesh_cam{:02d}_t{:.3f}.gif'.format(i, threshold)
            f_movie = meshsavedir + '/lego_mesh_cam{:02d}_t{:.3f}.mp4'.format(i, threshold)
            imageio.mimwrite(f_gif, imgs, duration=0.2)
            imageio.mimwrite(f_movie, imgs, fps=8)

        exit(1)

    # ############# write point cloud  ################
    if 0:
        print('i_val_internal:',i_val_internal)
        print('i_val_external:',i_val_external)
        i_test = i_val_internal + i_val_external
        i_val_external = [31]
        for i, img_i in enumerate(tqdm(i_val_external)):
            t = time.time()

            # Random from one image
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            #####  Core optimization loop  #####
            with torch.no_grad():
                rgbs, disps, pts, alpha, rgb_mean = render_path(pose, hwf, args.chunk, render_kwargs_test) # rgbs, (N, H, W, 3, k3)
                
    
            testsavedir = os.path.join(args.basedir, args.dataname, args.type_flows, args.expname, 'testset_{:06d}'.format(step))
            os.makedirs(testsavedir, exist_ok=True)

            pts = np.squeeze(pts).reshape(-1,3) #  (N, 3)
            alpha = alpha.squeeze() #  (H,W,K)
            rgb_mean = np.squeeze(rgb_mean).reshape(-1,3) # (N, 3)
            

            for k in range(rgbs.shape[-1]):
                alpha_show = alpha[...,k].reshape(-1)

                threshold = 1.
                ind = np.where(alpha_show > threshold)
                pts_show = pts[ind]
                pts_show[...,-1] *= -1
                rgb_show = rgb_mean[ind]

                print(pts.shape)
                print(pts_show.shape)

                pointcloud_dir = os.path.join(testsavedir, 'pointcloud')
                os.makedirs(pointcloud_dir, exist_ok=True)
                write_pointcloud(pointcloud_dir+'/{:02d}_t{:.3f}_k{:02d}_kl_pointcloud.ply'.format(img_i, threshold, k), pts_show, to8b(rgb_show))
                print('finish no.', k)
                exit(1)

            print(i, time.time() - t)

            exit(1)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    if args.is_train:
        train(args)
    else:
        test(args)   