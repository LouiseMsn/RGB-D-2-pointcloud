# RGB-D based point cloud generation with object recognition

## Installation

```bash
# 1. Clone this repo
git clone git@github.com:LouiseMsn/RGB-D-2-pointcloud.git
cd RGB-D-2-pointcloud

# 2. Clone the dependencies
mkdir third-party && cd third-party
git clone git@github.com:luca-medeiros/lang-segment-anything.git
cd ..

# 3. Create the conda environment
conda env create -f environment.yml 
conda activate rgbd-pointcloud
```

## Usage
This code uses a realsense 345 make sure it is plugged in and launch:
```bash
python pose-estimation.py
```

# TODO
- [ ] rename exec
- [ ] name repo
- [ ] fix color in pointcloud
