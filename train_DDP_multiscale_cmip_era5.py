import os
import math
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
from GSSAViT_multiscale_cmip_era5 import SuperFormer_v2
from dataset_cmip_era5 import CMIP6_ERA5PointCloudDataset
from scene.gaussian_model_cf_channel5 import CF3DGS_Render as GS_Render
from trainer.trainer_channel5 import GaussianTrainer
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid
from loss import L2_LOSS

class InfiniteDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    def __iter__(self):
        while True:
            # Use the parent's __iter__ method to get the indices for this process
            indices = list(super().__iter__())
            # Yield the indices in an infinite loop
            yield from indices


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_iterations", type=int, default=200001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--chkp_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=1)
    return parser.parse_args()


def get_dataset():
    cfg = {
        'pressure': {'z': [500.], 't': [850.],},
        'single': ['v10', 'u10', 't2m',]
    }

    time_ranges = [("1979-01-01 00:00:00", "2010-12-31 18:00:00")]
    return CMIP6_ERA5PointCloudDataset(cfg, time_ranges)


def load_era5_stats():
    with open('./era5_stats.json', 'r') as f:
        stats = json.load(f)
    cfg = {
        'pressure': {'z': [500.], 't': [850.],},
        'single': ['v10', 'u10', 't2m',]
    }
    max_list, min_list = [], []

    for vname, levels in cfg['pressure'].items():
        for level in levels:
            key = f"{vname}_{level}"
            if key in stats:
                max_list.append(stats[key]["avg_max"])
                min_list.append(stats[key]["avg_min"])

    for vname in cfg['single']:
        if vname in stats:
            max_list.append(stats[vname]["avg_max"])
            min_list.append(stats[vname]["avg_min"])
            
    max_list = np.array(max_list).reshape(1, 1, -1)
    min_list = np.array(min_list).reshape(1, 1, -1)
    
    return max_list, min_list

def load_era5_mean_std():
        with (open('./mean_std.json', 'r') as f1,
            open('./mean_std_single.json', 'r') as f2):
            stats_pressure = json.load(f1)
            stats_single = json.load(f2)
        cfg = {
            'pressure': {'z': [500.], 't': [850.],},
            'single': ['v10', 'u10', 't2m',]
        }
        mean_list, std_list = [], []

        total_levels = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
                        775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
                        350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
                        70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
                        1.]

        for vname, levels in cfg['pressure'].items():
            for level in levels:
                idx = total_levels.index(level)
                mean_list.append(stats_pressure['mean'][vname][idx])
                std_list.append(stats_pressure['std'][vname][idx])

        for vname in cfg['single']:
            mean_list.append(stats_single['mean'][vname])
            std_list.append(stats_single['std'][vname])

        mean_list = np.array(mean_list).reshape(1, 1, -1)
        std_list = np.array(std_list).reshape(1, 1, -1)

        return mean_list, std_list

def latitude_weighted_rmse(obs, model, height):
    lats = np.linspace(-90, 90, height)
    lat_rad = np.deg2rad(lats)

    weights = np.cos(lat_rad)
    weights /= weights.mean()
    weights = weights[:, np.newaxis]
    

    weighted_se = (weights * (obs - model)**2).mean()

    wrmse = np.sqrt(weighted_se)
    return wrmse

def compute_all_wrmses(gt, outputs):
    cfg = {
        'pressure': {'z': [500.], 't': [850.],},
        'single': ['v10', 'u10', 't2m',]
    }

    var_names = []

    for vname, levels in cfg['pressure'].items():
        for level in levels:
            var_names.append(f"{vname}_{level}")

    for vname in cfg['single']:
        var_names.append(vname)

    wrmses = []

    for i in range(gt.shape[2]):
        wrmse = latitude_weighted_rmse(gt[:, :, i], outputs[:, :, i], gt.shape[0])
        wrmses.append((var_names[i], wrmse))

    return wrmses

def get_patch_centers(latitudes, longitudes, patch_size=(9,8), patch_stride=(8,8)):
    H, W = len(latitudes), len(longitudes)
    ph, pw = patch_size
    sh, sw = patch_stride
    
    center_lat_list = []
    center_lon_list = []
    patch_indices = []  

    for h in range(0, H - ph + 1, sh):
        for w in range(0, W - pw + 1, sw):
        
            center_h = h + ph // 2
            center_w = w + pw // 2
            
            center_lat = latitudes[center_h]
            center_lon = longitudes[center_w]

            center_lat_list.append(center_lat)
            center_lon_list.append(center_lon)
            patch_indices.append((h, w))

    return np.array(center_lat_list), np.array(center_lon_list), patch_indices


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    full_dataset = get_dataset()
    train_indices = [idx for idx, (pc_idx) in enumerate(full_dataset.sample_indices)
                     if pd.to_datetime(full_dataset.pc_timestamps[pc_idx]).year <= 2020]
    train_subset = Subset(full_dataset, train_indices)

    sampler = InfiniteDistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=(8 > 0)
    )

    from dataset_6hours import PointCloudLoader
    gs_data = PointCloudLoader('./sample-2019-01-01-00.ply').get_full_cloud()
    model = SuperFormer_v2(
        arch='vit_base',
        patch_size=(1, 1),
        patch_stride=(1, 1),
        in_chans=5,
        out_chans=5,
        pretrained_model=None,
        kwargs=dict(
            learnable_pos=True,
            window=False,
            window_size=[(12, 8), (24, 16), (36, 24), (48, 32),
                        (6, 16), (12, 32), (18, 48), (24, 64),
                        (24, 4), (48, 8), (72, 12), (96, 16),
                        (12, 8), (24, 16), (36, 24), (48, 32),
                        (6, 16), (12, 32), (18, 48), (24, 64),
                        (24, 4), (48, 8), (72, 12), (96, 16),],
            interval=100,
            drop_path_rate=0,
            round_padding=True,
            pad_attn_mask=True,
            test_pos_mode='learnable_simple_interpolate',
            lms_checkpoint_train=False,
            img_size=(32, 64),
        )
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    gs_render = GS_Render(white_background=False, view_dependent=True)

    logger = setup_logger(args.log_dir) if rank == 0 else None
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model Parameters: {total_params:,}")
        logger.info(f"Dataset Size: {len(full_dataset)} | Train Subset: {len(train_subset)}")
        logger.info(f"World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")


    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.total_iterations / 4 * 2), eta_min=1e-6)
    criterion = nn.L1Loss()
    

    start_iteration = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.module.load_state_dict(state_dict, strict=True)
        if 'criterion' in ckpt:
            criterion.load_state_dict(ckpt['criterion'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
              
        start_iteration = ckpt.get('iteration', -1) + 1
        if rank == 0:
            logger.info(f"Resumed from {args.resume_checkpoint} at iter {start_iteration}")

    for iteration, (gt, ori_weather, scale, H_gt, W_gt) in enumerate(train_loader, start=start_iteration):
        if iteration >= args.total_iterations:
            break
        latitudes = np.linspace(-90, 90, 32)
        longitudes = np.linspace(-180, 180, 64)
        center_lats, center_lons, patch_starts = get_patch_centers(
                                                                    latitudes, 
                                                                    longitudes, 
                                                                    patch_size=(1,1), 
                                                                    patch_stride=(1,1)
                                                                )
        center_lats_t = torch.tensor(center_lats, dtype=torch.float32).unsqueeze(0)
        center_lons_t = torch.tensor(center_lons, dtype=torch.float32).unsqueeze(0)
        gs_datax_min, gs_datax_max = gs_data[:, 0].min(), gs_data[:, 0].max()
        gs_datay_min, gs_datay_max = gs_data[:, 1].min(), gs_data[:, 1].max()
    
        x = (center_lons_t+180)  * (gs_datax_max - gs_datax_min) / 360 + gs_datax_min  # [-180, 180]
        y = (center_lats_t +90)  * (gs_datay_max - gs_datay_min) / 180 + gs_datay_min   # [-90, 90]
        z = torch.ones(gt.shape[0], center_lats_t.shape[1], 1)
        xyz = torch.cat([x[:,:,None], y[:,:,None], z], dim=2).to(device, dtype=torch.float32)
        
        gt = gt.to(device, non_blocking=True, dtype=torch.float32)
        ori_weather = ori_weather.to(device, non_blocking=True, dtype=torch.float32)


        optimizer.zero_grad()
        outputs, gt, x_interpolate = model(ori_weather.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), scale)#x_gaussian=inputs
 
        gs_outputs = torch.cat([xyz, outputs], dim=2)
        _, pcd, viewpoint_cam = GaussianTrainer.prepare_custom_data(gs_outputs.squeeze(0), H_gt, W_gt, orthogonal=True, down_sample=True)
        gs_render.reset_model()
        gs_render.init_model(pcd, device = device)
        render_pkg = gs_render.render(viewpoint_cam, compute_cov3D_python=False, convert_SHs_python=True, override_color=None)
        outputs = render_pkg["image"]#[160,721, 1440]
        outputs = outputs + x_interpolate.squeeze(0)
      
        loss = criterion(outputs.unsqueeze(0), gt)#gt.permute(0, 3, 1, 2)
        loss.backward()
        if rank == 0 and iteration % 2000 == 0:  
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.info(f"{name} grad norm: {param.grad.norm().item():.6f}")
                else:
                    logger.info(f"{name} grad is None")

        optimizer.step()
        if iteration <= (args.total_iterations / 4 * 2):
            scheduler.step()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.eta_min

        if rank == 0 and iteration % args.log_interval == 0:
            logger.info(
                f"Iter {iteration:6d}/{args.total_iterations} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

        if rank == 0 and iteration % 500 == 0:
            gt = gt.squeeze(0)   
            gt = gt.detach().cpu().permute(1, 2, 0).numpy()
            outputs = outputs.detach().cpu().permute(1, 2, 0).numpy()
            max_values, min_values = load_era5_stats()
            mean_values, std_values = load_era5_mean_std()

            gt = gt * std_values + mean_values
            outputs = outputs * std_values + mean_values
            wrmses = compute_all_wrmses(gt, outputs)
            for name, wrmse in wrmses:
                logger.info(f"{name}: {wrmse:.8f}")
            logger.info(f"Number of points {center_lats_t.shape[1]}")
            logger.info(f"Scale {scale[0]}")

        if iteration % 10000 == 0 and iteration > 40000 and rank == 0:
            os.makedirs(args.chkp_dir, exist_ok=True)
            ckpt_path = os.path.join(args.chkp_dir, f"iter_{iteration}.pth")
           
            torch.save({
                'model': model.module.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': iteration
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
