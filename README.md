<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Generative 3D Gaussian Splatting for Arbitrary-Resolution Atmospheric Downscaling and Forecasting</h1>
  <!-- <p align="center">
    <a><strong>Zhibin Wen</strong></a>
    ·
    <a><strong>Tao Han</strong></a>
    ·
    <a><strong>Zhenghao Chen</strong></a>
    · -->

## Installation

##### (Recommended)
The codes have been tested on python 3.10, CUDA>=11.8. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

Adjust ```NUM_CHANNELS``` in ```/submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h``` to set the number of weather variables for reconstruction.

```
conda create -n wea3dgs python=3.10
conda activate wea3dgs
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone --recursive https://github.com/binbin2xs/weather-GS.git
pip install -r requirements.txt
```

## Dataset Preparsion
Set the data path in /trainer/trainer.py (line 51). 

For example:
```/dataset/era5_np_float32_part/2020/2020-01-01/00:00:00-t-1.0.npy``` represents:

Timestamp: 00:00:00 on 2020-01-01

Variable: Temperature (t) at pressure level 1.0

Shape: 721×1440

## Run （Per-sample 3DGS reconstruction）

### Training

Single-GPU reconstruction command:
```
python run_weather_recon.py --sh_degree 0 --start_time [start_time(YYYY-MM-DD)] --end_time [end_time(YYYY-MM-DD)]
```

Multi-GPU reconstruction command 
```
bash run_weather_recon.sh
or
python run_weather_recon_parralel.py --gpus 0-15 --gl_start_time [start_time(YYYY-MM-DD)] --gl_end_time [end_time(YYYY-MM-DD)]
```

### Render
If you want to directly render an image (.npy file) from a point cloud (.ply file), you can run the following command:
```
python render_weather.py
```
Image Resolution: Adjust at Line 172&173.

Point Cloud Input Path: Set at Line 286.

Rendered Image Output Path: Set at Line 297.



## Run (Neural Network-based generative 3D Gaussian Splatting framework for atmospheric downscaling and forecasting)

### Training

We provide different training scripts for different downscaling and forecasting settings.

#### MPI-ESM 5.625° to ERA5 1.40625°

For the downscaling task from MPI-ESM at 5.625° resolution to ERA5 at 1.40625° resolution, run:

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<MASTER_PORT> train_DDP_multiscale_cmip_era5.py
```

#### ERA5 5.625° to ERA5 2.8125°

For the ERA5-to-ERA5 fixed-resolution downscaling task from 5.625° to 2.8125°, run:

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<MASTER_PORT> train_DDP_fixscale_era5_era5.py
```

#### ERA5 1.40625° to ERA5 0.703125°

For the arbitrary-resolution forecasting task from ERA5 at 1.40625° resolution to ERA5 at 0.703125° resolution, run:

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<MASTER_PORT> train_DDP_multiscale.py
```

## Baselines

For fair comparison, the data preprocessing follows the same setting as [MINet](https://github.com/Teenye/MINet/). The reproduced baseline code can be found in the official [MINet](https://github.com/Teenye/MINet/) repository.



## Acknowledgement
Our recon is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [CF3DGS](https://github.com/NVlabs/CF-3DGS/tree/main). 
We thank all the authors for their great repos.


