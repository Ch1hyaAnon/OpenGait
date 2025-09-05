# 3D Pose Gait Recognition

This directory contains the implementation of **ThreeDPoseGait**, a 3D pose-based gait recognition model derived from SkeletonGait++.

## Overview

ThreeDPoseGait extends the SkeletonGait++ architecture to work with 3D pose data instead of 2D pose data. The key difference is the use of 3D coordinate information (x, y, z) rather than just 2D coordinates (x, y).

## Key Modifications

The following changes were made from the original SkeletonGait++ implementation:

### 1. Input Channel Configuration
- **Original**: 2 channels for 2D pose coordinates (x, y)
- **ThreeDPoseGait**: 3 channels for 3D pose coordinates (x, y, z)

### 2. Network Architecture
```python
# Original SkeletonGait++
self.map_layer0 = SetBlockWrapper(nn.Sequential(
    conv3x3(2, self.inplanes, 1),  # 2 channels for 2D pose
    nn.BatchNorm2d(self.inplanes),
    nn.ReLU(inplace=True)
))

# ThreeDPoseGait
self.map_layer0 = SetBlockWrapper(nn.Sequential(
    conv3x3(3, self.inplanes, 1),  # 3 channels for 3D pose
    nn.BatchNorm2d(self.inplanes),
    nn.ReLU(inplace=True)
))
```

### 3. Input Processing
```python
# Original: Extract 2 channels (x, y)
maps = pose[:, :2, ...]  

# ThreeDPoseGait: Extract 3 channels (x, y, z)
maps = pose[:, :3, ...]  
```

### 4. Input Dimension Assertions
- **Original**: Expected dimensions [44, 48, 88, 96] for 2D pose
- **ThreeDPoseGait**: Expected dimensions [66, 72, 132, 144] for 3D pose (approximately 1.5x)

### 5. Data Preprocessing
The input preprocessing now handles 4-channel data: 3 channels for 3D pose + 1 channel for silhouette.

## Configuration Files

Two configuration files are provided:

1. **3Dposegait_SUSTech1K.yaml** - Configuration for SUSTech1K dataset
2. **3Dposegait_Gait3D.yaml** - Configuration for Gait3D dataset

Both configurations specify:
- `model: ThreeDPoseGait`
- `in_channels: 4` (3 for 3D pose + 1 for silhouette)

## Usage

### Training
```bash
# For SUSTech1K dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/3Dposegait/3Dposegait_SUSTech1K.yaml --phase train

# For Gait3D dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/3Dposegait/3Dposegait_Gait3D.yaml --phase train
```

### Testing
```bash
# For SUSTech1K dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/3Dposegait/3Dposegait_SUSTech1K.yaml --phase test

# For Gait3D dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/3Dposegait/3Dposegait_Gait3D.yaml --phase test
```

## Data Format

The model expects input data in the following format:
- **3D Pose**: 3 channels representing (x, y, z) coordinates of pose keypoints
- **Silhouette**: 1 channel representing the silhouette mask
- **Total**: 4 channels per frame

The data should be structured as: `[T, 4, H, W]` where:
- `T` = temporal dimension (number of frames)
- `4` = channels (3 for 3D pose + 1 for silhouette)
- `H, W` = spatial dimensions (height, width)

## File Structure

```
configs/3Dposegait/
├── 3Dposegait_SUSTech1K.yaml    # SUSTech1K dataset configuration
└── 3Dposegait_Gait3D.yaml       # Gait3D dataset configuration

opengait/modeling/models/
└── threedposegait.py             # ThreeDPoseGait model implementation
```

## Relationship to SkeletonGait++

ThreeDPoseGait maintains the same overall architecture as SkeletonGait++:
- Dual-branch processing (pose + silhouette)
- Attention-based fusion mechanism
- Temporal pooling and horizontal pyramid matching
- Same loss functions and training procedures

The only difference is the enhanced capability to process 3D pose information, making it suitable for scenarios where depth information is available and can improve gait recognition performance.