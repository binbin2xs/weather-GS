import os
import sys
import math
import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement

from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View3
from utils.graphics_utils import focal2fov, fov2focal
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


import pdb


class Camera(nn.Module):
    def __init__(self, width, height, R, T, FoVx, FoVy, intrinsics=None,
                 trans=np.array([0.0, 0.0, 0.0], dtype=np.float32), scale=1.0, data_device="cuda",
                 do_grad=False, is_co3d=False):
        super(Camera, self).__init__()

        self.R = R.astype(np.float32)
        self.T = T.astype(np.float32)
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.intrinsics = intrinsics.astype(np.float32)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = int(width)
        self.image_height = int(height)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.astype(np.float32)
        self.scale = float(scale)

        self.world_view_transform = torch.tensor(getWorld2View3(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        w, h = self.image_width, self.image_height
        fx, fy, cx, cy = self.intrinsics[0, 0], self.intrinsics[1, 1], \
                         self.intrinsics[0, 2], self.intrinsics[1, 2]
        far, near = self.zfar, self.znear
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                  [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                  [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                  [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32).cuda().transpose(0, 1)
        self.projection_matrix = opengl_proj
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class CFGaussianModel:
    def __init__(self, sh_degree: int, view_dependent=True):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0, dtype=torch.float32).cuda()
        self._features_dc = torch.empty(0, dtype=torch.float32).cuda()
        self._scaling = torch.empty(0, dtype=torch.float32).cuda()
        self._rotation = torch.empty(0, dtype=torch.float32).cuda()
        self._opacity = torch.empty(0, dtype=torch.float32).cuda()
        self.max_radii2D = torch.empty(0, dtype=torch.float32).cuda()
        self.xyz_gradient_accum = torch.empty(0, dtype=torch.float32).cuda()
        self.denom = torch.empty(0, dtype=torch.float32).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.rotate_xyz = False
        self.rotate_seq = False
        self.seq_idx = 0
        self.view_dependent = view_dependent
    
    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        xyz = self._xyz.clone()
        if self.rotate_xyz:
            xyz = self.P[0].retr().act(xyz)
            return xyz
        elif self.rotate_seq:
            xyz = self.P[self.seq_idx].retr().act(xyz)
            return xyz
        else:
            return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        features_dc = self._features_dc
        return features_dc

    def get_RT(self, idx=None):
        if getattr(self, "P", None) is None:
            return torch.eye(4, dtype=torch.float32, device="cuda")

        if self.rotate_xyz:
            Rt = self.P[0].retr().matrix()
        else:
            if idx is None:
                Rt = self.P[self.seq_idx].retr().matrix()
            else:
                Rt = self.P[idx].retr().matrix()

        return Rt.squeeze()
    
    def load_ply(self, path, num_gauss=-1):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"], dtype=np.float32),
                       np.asarray(plydata.elements[0]["y"], dtype=np.float32),
                       np.asarray(plydata.elements[0]["z"], dtype=np.float32)), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 160, 1), dtype=np.float32)
        for i in range(160):
            features_dc[:, i, 0] = np.asarray(plydata.elements[0][f"f_dc_{i}"], dtype=np.float32)

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)

        if num_gauss == -1:
            num_gauss = xyz.shape[0]
        self._xyz = nn.Parameter(torch.tensor(
            xyz[:num_gauss], dtype=torch.float32, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(
            features_dc[:num_gauss], dtype=torch.float32, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(
            opacities[:num_gauss], dtype=torch.float32, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales[:num_gauss], dtype=torch.float32, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots[:num_gauss], dtype=torch.float32, device="cuda").requires_grad_(True))

        R = torch.eye(3, dtype=torch.float32, device="cuda")
        T = torch.zeros(1, 3, dtype=torch.float32, device="cuda")
        self.R = nn.Parameter(R.requires_grad_(True))
        self.T = nn.Parameter(T.requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

def prepare_custom_data():
    width = 1440
    height = 721

    fov = 79.0
    FoVx = fov * math.pi / 180
    intr_mat = np.eye(3, dtype=np.float32)
    intr_mat[0, 0] = fov2focal(FoVx, width)
    intr_mat[1, 1] = fov2focal(FoVx, width)
    intr_mat[0, 2] = width / 2
    intr_mat[1, 2] = height / 2           
    intr_mat[:2, :] /= 2

    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    focal_length_x = intr_mat[0, 0]
    focal_length_y = intr_mat[1, 1]
    FoVy = focal2fov(focal_length_y, height)
    FoVx = focal2fov(focal_length_x, width)

    viewpoint_camera = Camera(width, height, R, t, FoVx, FoVy,
                            intrinsics=intr_mat,
                            is_co3d=True)

    return viewpoint_camera

def render(gaussians, viewpoint_camera, scaling_modifier=1.0,
           compute_cov3D_python=False, convert_SHs_python=False):
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz,
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg_color = torch.tensor(
        [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_python:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if convert_SHs_python:
            shs_view = gaussians.get_features.transpose(1, 2).view(
            -1, 160, (gaussians.max_sh_degree + 1) ** 2
            )
            camera_center = torch.zeros(3, dtype=torch.float32, device="cuda")
            camera_center = camera_center[None].repeat(
            gaussians.get_features.shape[0], 1)
            dir_pp = gaussians._xyz - camera_center
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
            gaussians.active_sh_degree, shs_view, dir_pp_normalized
            )
            colors_precomp = sh2rgb + 0.5

    out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
    )
    rendered_image, _, _, _ = out

    return rendered_image



gaussians = CFGaussianModel(sh_degree=0)
gaussians.load_ply("2020-12-10 00:00:00.ply")

viewpoint_cam = prepare_custom_data()

image = render(gaussians,
            viewpoint_cam,
            compute_cov3D_python=False,
            convert_SHs_python=True,
            )
        
image = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
np.save('2020-12-10 00:00:00.npy', image)


