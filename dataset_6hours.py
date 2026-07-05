import os
import json
import numpy as np
import pandas as pd
from plyfile import PlyData
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import pdb
# try:
from s3_client import s3_client
# except:
#     s3_client = None
    

class PointCloudLoader:
    def __init__(self, ply_path):
        self._xyz, self._features_dc, self._opacity, self._scaling, self._rotation = self._load_ply(ply_path)
        
    def _load_ply(self, path):
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"], dtype=np.float32),
                       np.asarray(plydata.elements[0]["y"], dtype=np.float32),
                       np.asarray(plydata.elements[0]["z"], dtype=np.float32)), axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 160, 1), dtype=np.float32)
        for i in range(160):
            features_dc[:, i, 0] = np.asarray(plydata.elements[0][f"f_dc_{i}"], dtype=np.float32)
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name], dtype=np.float32)
        
        num_gauss = xyz.shape[0]
        xyz = torch.tensor(
            xyz[:num_gauss], dtype=torch.float32).detach().requires_grad_(False)
        features_dc = torch.tensor(
            features_dc[:num_gauss], dtype=torch.float32).transpose(1, 2).contiguous().detach().requires_grad_(False)
        opacities = torch.tensor(
            opacities[:num_gauss], dtype=torch.float32).detach().requires_grad_(False)
        scales = torch.tensor(
            scales[:num_gauss], dtype=torch.float32).detach().requires_grad_(False)
        rots = torch.tensor(
            rots[:num_gauss], dtype=torch.float32).detach().requires_grad_(False)

        return xyz, features_dc, opacities, scales, rots
    
    
    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_features(self):
        features_dc = self._features_dc.squeeze(1)
        return features_dc
    
    def get_full_cloud(self):
        xyz = self.get_xyz
        features = self.get_features
        opacity = self.get_opacity
        scaling = self.get_scaling
        rotation = self.get_rotation
        return torch.cat([xyz, features, opacity, scaling, rotation], dim=1)  #(N, 171)




class ERA5PointCloudDataset(Dataset):
    def __init__(self, cfg, time_ranges, point_cloud_root, data_root_dir="/project/peilab/dataset/era5_np_float32_part"):
        self.cfg = cfg
        self.data_root_dir = data_root_dir
        self.point_cloud_root = point_cloud_root
        self.executor = ThreadPoolExecutor(max_workers=16)

        all_timestamps = []
        for start, end in time_ranges:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            ts_range = pd.date_range(start=start_dt, end=end_dt, freq='1h')
            all_timestamps.append(ts_range)
        
        self.pc_timestamps = pd.DatetimeIndex(np.concatenate(all_timestamps))
        
        self.point_cloud_paths = []
        
        self.max_values, self.min_values = self._load_era5_stats()
        self.mean_values, self.std_values = self._load_era5_mean_std()

        num_pairs = len(self.pc_timestamps) - 6
        self.sample_indices = [(i, i + 6) for i in range(num_pairs)]

        bucket =dict(endpoint ='your_endpoint_url',)
        self.load_data_from_ceph = s3_client(**bucket) if s3_client else None
        
    def _load_era5_frame(self, timestamp):
        file_paths = []

        for vname in self.cfg['vnames']['pressure']:
            for height in self.cfg['pressure_level']:
                path = os.path.join(
                    self.data_root_dir,
                    timestamp[:4],
                    timestamp[:10],
                    f"{timestamp[-8:]}-{vname}-{height}.npy"
                )
                file_paths.append(path)

        for vname in self.cfg['vnames']['single']:
            path = os.path.join(
                self.data_root_dir, "single",
                timestamp[:4], timestamp[:10],
                f"{timestamp[-8:]}-{vname}.npy"
            )
            file_paths.append(path)
        
        data = []
        for path in file_paths:
            arr = np.load(path)
            if 'tp' in path:
                arr *= 1000  # Convert precipitation units
            data.append(arr)
        data = np.stack(data, axis=-1)
        
        return data
    
    def _load_data(self, timestamp):
        file_paths = []

        for vname in self.cfg['vnames']['pressure']:
            for height in self.cfg['pressure_level']:
                idx = f'/{timestamp.strftime("%Y")}/{timestamp.strftime("%Y-%m-%d")}/{timestamp.strftime("%H:%M:%S")}-{vname}-{height}.npy'
                file_paths.append((idx, 'pressure'))
        for vname in self.cfg['vnames']['single']:
            idx = f'/single/{timestamp.strftime("%Y")}/{timestamp.strftime("%Y-%m-%d")}/{timestamp.strftime("%H:%M:%S")}-{vname}.npy'
            file_paths.append((idx, 'single'))

        def load_file(file_info):
            idx, var_type = file_info
            vdata = self.load_data_from_ceph.read_npy_from_BytesIO(objectName=idx)
            vdata = vdata[None, :, :]
            if var_type == 'single' and 'tp' in idx:
                vdata = vdata * 1000
            return vdata

        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(self.executor.map(load_file, file_paths))

            input_initial_field = np.concatenate(results, axis=0).transpose(1,2,0)
        
        return input_initial_field


    def _load_era5_stats(self):
        with open('./era5_stats.json', 'r') as f:
            stats = json.load(f)
        
        max_list, min_list = [], []

        for vname in self.cfg['vnames']['pressure']:
            for level in self.cfg['pressure_level']:
                key = f"{vname}_{level}"
                if key in stats:
                    max_list.append(stats[key]["avg_max"])
                    min_list.append(stats[key]["avg_min"])
    
        for vname in self.cfg['vnames']['single']:
            if vname in stats:
                max_list.append(stats[vname]["avg_max"])
                min_list.append(stats[vname]["avg_min"])
        
        return torch.tensor(max_list), torch.tensor(min_list)

    def _normalize_era5(self, data):
        data_tensor = torch.from_numpy(data).float()
        return (data_tensor - self.min_values) / (self.max_values - self.min_values + 1e-8)

    def _load_era5_mean_std(self):
        with (open('./mean_std.json', 'r') as f1,
            open('./mean_std_single.json', 'r') as f2):
            stats_pressure = json.load(f1)
            stats_single = json.load(f2)

        mean_list, std_list = [], []

        total_levels = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
                        775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
                        350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
                        70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
                        1.]

        for vname in self.cfg['vnames']['pressure']:
            if vname not in stats_pressure['mean'] or vname not in stats_pressure['std']:
                raise KeyError(f"Missing variable {vname} in pressure stats")

            for level in self.cfg['pressure_level']:
                try:
                    idx = total_levels.index(level)
                except ValueError:
                    raise ValueError(f"Level {level} not found in total_levels")

                mean_list.append(stats_pressure['mean'][vname][idx])
                std_list.append(stats_pressure['std'][vname][idx])

        for vname in self.cfg['vnames']['single']:
            if vname not in stats_single['mean'] or vname not in stats_single['std']:
                raise KeyError(f"Missing variable {vname} in single stats")

            mean_list.append(stats_single['mean'][vname])
            std_list.append(stats_single['std'][vname])

        return torch.tensor(mean_list), torch.tensor(std_list)

    def _standardize_era5(self, data):
        data_tensor = torch.from_numpy(data).float()
        return (data_tensor - self.mean_values) / (self.std_values)

    
    def __len__(self):
        return len(self.sample_indices)
    
    

    def get_future_frame(self, pc_idx, step):
        future_idx = pc_idx + step * 6  
        timestamp = self.pc_timestamps[future_idx]

        data = self._load_data(timestamp)
        data = self._normalize_era5(data)

        data[:,:,5] *= 2.0
        data[:,:,82] *= 1.4
        data[:,:,85] *= 1.4

        return data


    def __getitem__(self, idx):
        pc_idx, gt_idx = self.sample_indices[idx]

        initial = self._load_data(self.pc_timestamps[pc_idx])
        gt = self._load_data(self.pc_timestamps[gt_idx])
        initial = self._normalize_era5(initial)
        gt = self._normalize_era5(gt)
        
        return gt, initial, pc_idx
