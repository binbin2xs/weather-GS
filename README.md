<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">3D Gaussian Splatting for Weather Recon</h1>
  <p align="center">
    <a><strong>Zhibin Wen</strong></a>
    ·
    <a><strong>Tao Han</strong></a>
    ·
    <a><strong>Zhenghao Chen</strong></a>
    ·

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

## Run

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
## Acknowledgement
Our recon is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [CF3DGS](https://github.com/NVlabs/CF-3DGS/tree/main). 
We thank all the authors for their great repos.
