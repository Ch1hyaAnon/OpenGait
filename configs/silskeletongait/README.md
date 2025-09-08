# SilSkeletonGait Model

A novel gait recognition model that fuses **Silhouette** and **3D Skeleton** data, inspired by SkeletonGaitPP architecture.

## Overview

SilSkeletonGait addresses the requirement to combine silhouette images and 3D skeleton pose data for improved gait recognition performance. The model processes both modalities through dedicated branches and fuses them using an attention mechanism.

## Architecture

```
Input: 
├── Silhouette: (sequence_length, height, width)
└── 3D Skeleton: (sequence_length, 16_joints, 3_coords)

Processing:
├── Silhouette Branch: sil_layer0 → sil_layer1
├── Skeleton Branch: skeleton_to_heatmaps → skeleton_layer0 → skeleton_layer1  
└── Fusion: AttentionFusion(sil_features, skeleton_features)

Shared Layers:
layer2 → layer3 → layer4 → TemporalPooling → HorizontalPooling → FC → BNNecks
```

## Key Features

- **Dual-Modal Input**: Processes both silhouette images and 3D skeleton coordinates
- **3D to 2D Conversion**: Converts 3D skeleton data to 2D heatmaps for CNN processing
- **Attention Fusion**: Uses attention mechanism to optimally combine features from both modalities
- **OpenGait Compatible**: Fully integrated with OpenGait training and evaluation framework

## Data Format

### Input Data Structure
```python
# Silhouette data: numpy array
silhouette.shape = (sequence_length, height, width)  # e.g., (97, 64, 64)

# Skeleton data: dict containing 'result' key
skeleton = {
    'result': numpy_array  # shape: (sequence_length, 16, 3)
}
```

### Data Processing Pipeline
1. **Silhouette**: Add channel dimension → (T, 1, H, W)
2. **Skeleton**: Convert 3D coordinates to 2D heatmaps → (T, 2, H, W)
3. **Combine**: Concatenate → (T, 3, H, W) format compatible with SkeletonGaitPP

## Implementation Details

### Skeleton to Heatmap Conversion
```python
def _skeleton_to_maps(self, skeleton, H, W):
    """Convert 3D skeleton data to 2D feature maps"""
    # Create two channels:
    # Channel 0: Joint position heatmap (x,y coordinates)  
    # Channel 1: Depth information (z coordinates)
    
    # Normalize coordinates to image space [0, H-1] × [0, W-1]
    # Fill heatmaps with joint positions and depth values
```

### Model Components
- **Silhouette Branch**: `sil_layer0` + `sil_layer1` (BasicBlock2D)
- **Skeleton Branch**: `skeleton_layer0` + `skeleton_layer1` (BasicBlock2D)  
- **Fusion Layer**: `AttentionFusion` with squeeze-and-excitation style attention
- **Shared Backbone**: P3D blocks + temporal/spatial pooling + classification heads

## Usage

### Training
```bash
python opengait/main.py \
    --cfgs ./configs/silskeletongait/silskeletongait_sample.yaml \
    --phase train
```

### Configuration
```yaml
data_cfg:
  dataset_name: sample
  data_in_use: [True, True]  # [silhouette, skeleton]

model_cfg:
  model: SilSkeletonGait
  Backbone:
    in_channels: 3  # 2 skeleton channels + 1 silhouette channel
    blocks: [1, 4, 4, 1]
    C: 2
```

## File Structure

```
configs/silskeletongait/
└── silskeletongait_sample.yaml          # Configuration file

opengait/modeling/models/
└── silskeletongait.py                   # Model implementation

datasets/sample/
├── sil/seq0/sample.pkl                  # Silhouette data
└── skeleton/seq0/sample.pkl             # 3D skeleton data
```

## Technical Specifications

### Model Parameters
- **Input Channels**: 3 (2 skeleton + 1 silhouette)
- **Backbone**: ResNet-style with P3D blocks
- **Fusion**: Attention-based feature fusion
- **Output**: Identity embeddings + classification logits

### Data Requirements
- **Silhouette**: Grayscale images, shape (T, H, W)
- **Skeleton**: 3D joint coordinates, shape (T, 16, 3)
- **Synchronization**: Both modalities must have same sequence length T

### Performance Characteristics
- **Memory Efficient**: Converts sparse 3D skeleton to dense 2D heatmaps
- **Attention Fusion**: Learns optimal combination of modalities
- **Framework Compatible**: Works with existing OpenGait training pipeline

## Testing

Run the validation script to verify model functionality:
```bash
python /tmp/test_silskeletongait.py
```

Expected output:
```
✅ SilSkeletonGait Model Test SUCCESSFUL!
   The model can successfully:
   • Load both silhouette and 3D skeleton data
   • Convert 3D skeleton coordinates to 2D heatmaps  
   • Combine modalities for fusion-based processing
   • Integrate with OpenGait training framework
```

## Comparison with SkeletonGaitPP

| Aspect | SkeletonGaitPP | SilSkeletonGait |
|--------|----------------|-----------------|
| Input 1 | Heatmap (2D pose) | 3D Skeleton |
| Input 2 | Silhouette | Silhouette |
| Processing | Direct CNN | 3D→2D conversion + CNN |
| Fusion | Attention | Attention (same) |
| Framework | OpenGait | OpenGait |

## Future Improvements

1. **Advanced 3D Processing**: Direct 3D convolutions for skeleton data
2. **Multi-Scale Fusion**: Fusion at multiple network levels  
3. **Temporal Modeling**: Enhanced temporal relationship modeling
4. **Joint-Specific Attention**: Per-joint attention mechanisms

## References

- OpenGait: A Comprehensive Benchmark Study for Gait Recognition Towards Better Practicality
- SkeletonGaitPP: Learning Gait Representation from Skeleton Maps for Gait Recognition