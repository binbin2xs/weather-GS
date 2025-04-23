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

```bash
conda create -n wea3dgs python=3.10
conda activate wea3dgs
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Preparsion


## Run

### Training
```bash
python run_cf3dgs.py --sh_degree 0 --start_time 2020-01-02 --end_time 2020-01-03

bash /cpfs/hantao/CF-3DGS/run_weather_recon.sh

python run_cf3dgs_parralel.py --gpus 0-15 --gl_start_time 2008-01-01 --gl_end_time 2023-12-31
```


## Acknowledgement
Our recon is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [CF3DGS](https://github.com/NVlabs/CF-3DGS/tree/main). 
We thank all the authors for their great repos.
