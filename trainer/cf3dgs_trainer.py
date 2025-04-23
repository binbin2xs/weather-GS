# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from tqdm import tqdm
from random import randint
import math
import numpy as np
import random
from collections import defaultdict, OrderedDict
import json
import gzip
import torch
import torch.nn.functional as F
from torchvision import io
from PIL import Image
from einops import rearrange
import pickle
import scipy
import imageio
import glob
import cv2
import open3d as o3d

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model_cf import CF3DGS_Render as GS_Render

from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes
from scene.cameras import Camera
from utils.loss_utils import l1_loss, l2_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, colorize
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE

from kornia.geometry.depth import depth_to_3d, depth_to_normals
from kornia.geometry.camera import project_points

import pdb

from .trainer import GaussianTrainer
from .losses import Loss, compute_scale_and_shift

from copy import copy
from utils.vis_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose



class CFGaussianTrainer(GaussianTrainer):
    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        super().__init__(data_root, model_cfg, pipe_cfg, optim_cfg)
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg

        self.gs_render = GS_Render(white_background=False,
                                   view_dependent=model_cfg.view_dependent,)
        self.gs_render_local = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,)
        self.use_mask = self.pipe_cfg.use_mask
        self.use_mono = self.pipe_cfg.use_mono
        self.near = 0.01
        self.setup_losses()

    def setup_losses(self):
        self.loss_func = Loss(self.optim_cfg)

    def train_step(self,
                   gs_render,
                   viewpoint_cam,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   **kwargs,
                   ):
        # Render
        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = l1_loss(image, gt_image)
        loss.backward()

        with torch.no_grad():
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            
            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)

        return loss, render_pkg, psnr_train

    def init_two_view(self, view_idx, pipe, optim_opt, result_path):
        # prepare data
        self.loss_func.depth_loss_type = "invariant"
        cam_info, pcd, viewpoint_cam = self.prepare_data(view_idx,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()

        # Initialize gaussians
        self.gs_render.reset_model()
        self.gs_render.init_model(pcd,)
        print(f"optimizing frame {view_idx:03d}")
        optim_opt.iterations = self.single_step
        timestamp = self.timestamps[view_idx]
        year = timestamp[:4]
        gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc=f"Training progress{year}",
                            position=gpu_id)
        self.gs_render.gaussians.training_setup(optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=self.mono_depth[view_idx],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"LOSS": f"{loss:.{5}f}",
                                          "PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()
                
                os.makedirs(f"{result_path}/{optim_opt.iterations}it_{self.gs_render.gaussians.get_xyz.shape[0]}points/{year}", exist_ok=True)
                np.save(f"{result_path}/{optim_opt.iterations}it_{self.gs_render.gaussians.get_xyz.shape[0]}points/{year}/{timestamp}.npy",
                        rend_dict["image"].detach().cpu().permute(1, 2, 0).numpy())              
                self.gs_render.gaussians.save_ply(f"{result_path}/{optim_opt.iterations}it_{self.gs_render.gaussians.get_xyz.shape[0]}points/{year}/{timestamp}.ply")

    
    def train_from_progressive(self, ):
        pipe = copy(self.pipe_cfg)
        self.single_step = 2000

        if pipe.expname == "":
            expname = "recon"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        result_path = f"output/{expname}"
        os.makedirs(result_path, exist_ok=True)

        max_frame = self.seq_len
        start_frame = 1
        end_frame = max_frame

        num_eppch = 1
        reverse = False
        for fidx in range(0, end_frame):
            self.init_two_view(fidx, pipe, copy(self.optim_cfg), result_path)
