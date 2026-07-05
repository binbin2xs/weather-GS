import os
import json
import numpy as np
import pandas as pd
from plyfile import PlyData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import xarray as xr
import pdb
# try:
from s3_client import s3_client
# except:
#     s3_client = None
    


class CMIP6_ERA5PointCloudDataset(Dataset):
    def __init__(self, cfg, time_ranges):
        self.cfg = cfg

        all_timestamps = []
        for start, end in time_ranges:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            ts_range = pd.date_range(start=start_dt, end=end_dt, freq='6h')
            all_timestamps.append(ts_range)
        
        self.pc_timestamps = pd.DatetimeIndex(np.concatenate(all_timestamps))
        
        self.max_values, self.min_values = self._load_era5_stats()
        self.mean_values, self.std_values = self._load_era5_mean_std()

        self.sample_indices = list(range(len(self.pc_timestamps)))
                
        bucket_era5 =dict(endpoint ='your_endpoint_url',)
        self.load_data_from_ceph_era5 = s3_client(**bucket_era5) if s3_client else None

        bucket_cmip =dict(ak_sk = "your_ak_sk",
                         endpoint ='your_endpoint_url'
                  )
        self.load_data_from_ceph_cmip = s3_client(**bucket_cmip) if s3_client else None

        self.cmip_nc_cache = {} 

    def _get_cmip_dataset(self, idx, var_type):
        if idx in self.cmip_nc_cache:
            return self.cmip_nc_cache[idx]

        if var_type == 'pressure_local':
            ds = xr.open_dataset(idx)
        else:
            ds = self.load_data_from_ceph_cmip.read_nc_from_BytesIO(
                objectName=idx
            )
        ds.load()
        self.cmip_nc_cache[idx] = ds
        return ds

    def _load_data_era5(self, timestamp):
        file_paths = []

        for vname, levels in self.cfg['pressure'].items():
            for height in levels:
                idx = (
                    f'/{timestamp.strftime("%Y")}/'
                    f'{timestamp.strftime("%Y-%m-%d")}/'
                    f'{timestamp.strftime("%H:%M:%S")}-{vname}-{height}.npy'
                )
                file_paths.append((idx, 'pressure'))
        for vname in self.cfg['single']:
            idx = f'/{timestamp.strftime("%Y")}/{timestamp.strftime("%Y-%m-%d")}/{timestamp.strftime("%H:%M:%S")}-{vname}.npy'
            file_paths.append((idx, 'single'))

        def load_file_era5(file_info):
            idx, var_type = file_info
            vdata = self.load_data_from_ceph_era5.read_npy_from_BytesIO(objectName=idx)
            vdata = vdata[None, :, :]
            if var_type == 'single' and 'tp' in idx:
                vdata = vdata * 1000
            return vdata

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_file_era5, file_paths))

            input_initial_field = np.concatenate(results, axis=0).transpose(1,2,0)
        
        return input_initial_field

    def _get_cmip_time_suffix(self, timestamp):
        CMIP_TIME_RANGES = [
            (
                datetime(1970, 1, 1, 6),
                datetime(1990, 1, 1, 0),
                "197001010600-199001010000",
            ),
            (
                datetime(1990, 1, 1, 6),
                datetime(2010, 1, 1, 0),
                "199001010600-201001010000",
            ),
            (
                datetime(2010, 1, 1, 6),
                datetime(2015, 1, 1, 0),
                "201001010600-201501010000",
            ),
        ]
        for t_start, t_end, suffix in CMIP_TIME_RANGES:
            if t_start <= timestamp <= t_end:
                return suffix
        raise ValueError(f"Timestamp {timestamp} not covered by CMIP files")

    def _load_data_cmip(self, timestamp):
        arrays = []
        time_suffix = self._get_cmip_time_suffix(timestamp)

        CMIP_PRESSURE_MAP = {
            't': 'ta',
        }

        CMIP_SINGLE_MAP = {
            'u10': 'uas',
            'v10': 'vas',
            't2m': 'tas',
        }
        file_infos = []
        for vname, levels in self.cfg['pressure'].items():
            for level in levels:
                level_pa = int(level * 100)

                if vname == "z" and level == 500.0:
                    cmip_var = "zg500"
                    idx = (f"/CMIP6/CMIP/historical/MPI-ESM1-2-LR/6hrPlevPt/{cmip_var}/r1i1p1f1/gn/v20190815/{cmip_var}_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{time_suffix}.nc")
                    file_infos.append((idx, cmip_var, 'pressure_ceph', None))
                else:
                    cmip_var = CMIP_PRESSURE_MAP[vname]
                    idx = (
                        f"./output/cmip/"
                        f"{cmip_var}_plev{level_pa}_{time_suffix}.nc"
                    )
                    file_infos.append((idx, cmip_var, 'pressure_local', level))

        for vname in self.cfg['single']:
            cmip_var = CMIP_SINGLE_MAP[vname]
            if cmip_var == "tas":
                idx = (f"/CMIP6/CMIP/historical/MPI-ESM1-2-LR/6hrPlevPt/{cmip_var}/r1i1p1f1/gn/v20190815/{cmip_var}_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{time_suffix}.nc")
            else:
                idx = (f"/CMIP6/CMIP/historical/MPI-ESM1-2-LR/6hrPlevPt/{cmip_var}/r1i1p1f1/gn/v20190710/{cmip_var}_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{time_suffix}.nc")
            file_infos.append((idx, cmip_var, 'single', None))
           
        def load_file_cmip(file_info):
            idx, cmip_var, var_type, level = file_info
           
            ds = self._get_cmip_dataset(idx, var_type)
            da = ds[cmip_var].sel(time=timestamp)
            arr = da.values.astype(np.float32)
            if cmip_var == 'zg500':
                arr *= 9.80665
            if cmip_var == 'ta':
                arr = arr
            else:
                arr = arr[None, :, :]
            return arr

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_file_cmip, file_infos))

        cmip_field = np.concatenate(results, axis=0).transpose(1, 2, 0)
        return cmip_field

    def _load_era5_stats(self):
        with open('./era5_stats.json', 'r') as f:
            stats = json.load(f)
        
        max_list, min_list = [], []

        for vname, levels in self.cfg['pressure'].items():
            for level in levels:
                key = f"{vname}_{level}"
                if key in stats:
                    max_list.append(stats[key]["avg_max"])
                    min_list.append(stats[key]["avg_min"])
    
        for vname in self.cfg['single']:
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

        for vname, levels in self.cfg['pressure'].items():
            for level in levels:
                idx = total_levels.index(level)
                mean_list.append(stats_pressure['mean'][vname][idx])
                std_list.append(stats_pressure['std'][vname][idx])

        for vname in self.cfg['single']:
            mean_list.append(stats_single['mean'][vname])
            std_list.append(stats_single['std'][vname])

        return torch.tensor(mean_list), torch.tensor(std_list)

    def _standardize_era5(self, data):
        data_tensor = torch.from_numpy(data).float()
        return (data_tensor - self.mean_values) / (self.std_values)


    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        pc_idx = self.sample_indices[idx]
        
        era5_full = self._load_data_era5(self.pc_timestamps[pc_idx])#(721,1440,5)
        cmip_full = self._load_data_cmip(self.pc_timestamps[pc_idx])

        era5_full = self._standardize_era5(era5_full)
        cmip_full = self._standardize_era5(cmip_full)


        scale = np.random.uniform(1, 4 + 1e-8)

        cmip_tensor = cmip_full.permute(2,0,1).unsqueeze(0)
        cmip_lr = F.interpolate(cmip_tensor, size=(32,64), mode='bilinear', align_corners=False)
        cmip_lr = cmip_lr.squeeze(0).permute(1,2,0)

        H_gt, W_gt = int(32*scale), int(64*scale)
        era5_tensor = era5_full.permute(2,0,1).unsqueeze(0)
        era5_gt = F.interpolate(era5_tensor, size=(H_gt,W_gt), mode='bilinear', align_corners=False)
        era5_gt = era5_gt.squeeze(0).permute(1,2,0)

        scale = torch.tensor(scale, dtype=torch.float32)

        return era5_gt, cmip_lr, scale, H_gt, W_gt






if __name__ == "__main__":
    cfg = {
        'pressure': {
            'z': [500.],
            't': [850.],
        },
        'single': ["v10", "u10", "t2m"]
    }
    time_ranges = [
        ("1990-01-01 00:00:00", "1990-01-01 18:00:00")
    ]
    dataset = CMIP6_ERA5PointCloudDataset(cfg, time_ranges)
    print(f"Dataset length: {len(dataset)}")
    for idx in range(len(dataset)):
        era5_gt, cmip_lr, scale, H_gt, W_gt = dataset[idx]
        
        print(f"\nSample {idx}:")
        # print(f"ERA5 shape: {era5.shape}")
        # print(f"ERA5 min: {era5.min().item():.4f}, max: {era5.max().item():.4f}, mean: {era5.mean().item():.4f}")
        # print(f"CMIP shape: {cmip.shape}")
        # print(f"CMIP min: {cmip.min().item():.4f}, max: {cmip.max().item():.4f}, mean: {cmip.mean().item():.4f}")

        