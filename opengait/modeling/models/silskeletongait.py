import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, SetBlockWrapper, conv3x3, conv1x1, BasicBlock2D, BasicBlockP3D

from einops import rearrange

import copy

class SilSkeletonGait(BaseModel):

    def build_network(self, model_cfg):
        #B, C = [1, 4, 4, 1], 2
        in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
        self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        self.inplanes = 32 * C
        
        # Silhouette branch - same as SkeletonGaitPP
        self.sil_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        # Skeleton branch - process skeleton feature maps (similar to map_layer in SkeletonGaitPP)
        self.skeleton_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(2, self.inplanes, 1),  # 2 channels for skeleton feature maps
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        self.sil_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0], mode='2d'))
        self.skeleton_layer1 = copy.deepcopy(self.sil_layer1)
        self.fusion = AttentionFusion(32 * C)

        self.layer2 = self.make_layer(BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1], mode='p3d')
        self.layer3 = self.make_layer(BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2], mode='p3d')
        self.layer4 = self.make_layer(BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3], mode='p3d')

        self.FCs = SeparateFCs(16, 256*C, 128*C)
        self.BNNecks = SeparateBNNecks(16, 128*C, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def inputs_pretreament(self, inputs):
        ### Process silhouette and skeleton data
        sil_skeleton_data = inputs[0]
        new_data_list = []
        
        # Handle the case where we have [sil_list, skeleton_list]
        if len(sil_skeleton_data) == 2:
            for sil_data, skeleton_data in zip(sil_skeleton_data[0], sil_skeleton_data[1]):
                # Process silhouette: add channel dimension
                sil = sil_data[:, np.newaxis, ...]  # [T, 1, H, W]
                
                # Process skeleton: extract from dict if needed
                if isinstance(skeleton_data, dict):
                    skeleton = skeleton_data['result']  # [T, 16, 3]
                else:
                    skeleton = skeleton_data
                    
                # Convert 3D skeleton to 2D feature maps
                T, H, W = sil.shape[0], sil.shape[2], sil.shape[3]
                skeleton_maps = self._skeleton_to_maps(skeleton, H, W)  # [T, 2, H, W]
                
                # Concatenate: [skeleton_maps, silhouette] -> [T, 3, H, W]
                combined = np.concatenate([skeleton_maps, sil], axis=1)
                new_data_list.append(combined)
        else:
            # Fallback to original format
            new_data_list = sil_skeleton_data[0]
            
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)
    
    def _skeleton_to_maps(self, skeleton, H, W):
        """Convert 3D skeleton data to 2D feature maps"""
        T, num_joints, _ = skeleton.shape  # [T, 16, 3]
        
        # Create two feature maps: one for x-y coordinates, one for z coordinates
        maps = np.zeros((T, 2, H, W), dtype=np.float32)
        
        for t in range(T):
            # Normalize skeleton coordinates to image space
            joints = skeleton[t]  # [16, 3]
            
            # Assuming skeleton coordinates are in some world coordinate system
            # Normalize to [0, 1] range (simple approach)
            if np.any(joints != 0):
                # Normalize x, y coordinates to [0, H-1], [0, W-1]
                x_coords = joints[:, 0]
                y_coords = joints[:, 1] 
                z_coords = joints[:, 2]
                
                # Simple normalization (can be improved)
                if x_coords.max() > x_coords.min():
                    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
                else:
                    x_norm = np.ones_like(x_coords) * 0.5
                    
                if y_coords.max() > y_coords.min():
                    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
                else:
                    y_norm = np.ones_like(y_coords) * 0.5
                    
                if z_coords.max() > z_coords.min():
                    z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
                else:
                    z_norm = np.ones_like(z_coords) * 0.5
                
                # Convert to pixel coordinates
                x_pixels = (x_norm * (W - 1)).astype(int)
                y_pixels = (y_norm * (H - 1)).astype(int)
                
                # Fill the feature maps
                for joint_idx in range(num_joints):
                    x, y = x_pixels[joint_idx], y_pixels[joint_idx]
                    # Clamp to valid range
                    x = max(0, min(W-1, x))
                    y = max(0, min(H-1, y))
                    
                    # First channel: x-y position heatmap
                    maps[t, 0, y, x] = 1.0
                    # Second channel: z-depth information
                    maps[t, 1, y, x] = z_norm[joint_idx]
        
        return maps

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        pose = ipts[0]
        pose = pose.transpose(1, 2).contiguous()
        assert pose.size(-1) in [44, 48, 64, 88, 96]  # Allow 64x64 for our sample data
        
        # For our SilSkeletonGait: skeleton maps (2 channels) + silhouette (1 channel)
        skeleton_maps = pose[:, :2, ...]  # First 2 channels for skeleton
        sils = pose[:, -1, ...].unsqueeze(1)  # Last channel for silhouette

        del ipts
        
        # Process through skeleton branch
        skel0 = self.skeleton_layer0(skeleton_maps)
        skel1 = self.skeleton_layer1(skel0)
        
        # Process through silhouette branch
        sil0 = self.sil_layer0(sils)
        sil1 = self.sil_layer1(sil0)

        # Fusion
        out1 = self.fusion(sil1, skel1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        n, c, h, w = outs.size()

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        
        if self.inference_use_emb:
             embed = embed_2
        else:
             embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * 2), 
            )
        )
    
    def forward(self, sil_feat, skeleton_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            skeleton_feat: [n, c, s, h, w]
        '''
        c = sil_feat.size(1)
        feats = torch.cat([sil_feat, skeleton_feat], dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (d c) s h w -> n d c s h w', d=2)
        score = F.softmax(score, dim=1)
        retun = sil_feat * score[:, 0] + skeleton_feat * score[:, 1]
        return retun