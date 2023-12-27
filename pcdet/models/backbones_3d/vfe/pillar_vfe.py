import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import numpy as np
import pdb
import math
from .vfe_template import VFETemplate
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

def gen_pyramid_xyz(features, scale=0.01):
    """
    features shape [n_points,3]
    return 3 8-bit simulated quantized features with shape [n_points,3]
    """
    # int24 quantization on features
    scale_a = scale * (1 << 15)
    scale_b = scale * (1 << 8)
    scale_c = scale
    qmin = -128
    qmax = 127
    feature_a_int = (features / scale_a).round().clamp(qmin, qmax)
    feature_a_sim = feature_a_int * scale_a
    feature_b_int = ((features - feature_a_sim) / scale_b).round().clamp(qmin, qmax)
    feature_b_sim = feature_b_int * scale_b
    feature_c_int = ((features - feature_a_sim - feature_b_sim) / scale_c).round().clamp(qmin, qmax)
    feature_c_sim = feature_c_int * scale_c
    return feature_a_sim, feature_b_sim, feature_c_sim

def pillar_pyramid(features, k=1):
    """
    features shape [n_points,3]
    return 3 8-bit simulated quantized features with shape [n_points,3]
    """
    scale = 1.0
    scale_ls = []
    for i in range(k+1): #k=3
        scale_ls.append(scale * 2**i)
    scale_ls = sorted(scale_ls, reverse=True) #[8.0, 4.0, 2.0, 1.0]
    # qmin = -128
    # qmax = 127
    feats_ls = []
    res = features
    for j in range(len(scale_ls)):
        if j < k:
            x_j = (res / scale_ls[j]).round() #.clamp(qmin, qmax)
            res = res - x_j * scale_ls[j]
            feats_ls.append(x_j)
        else:
            feats_ls.append(res)
    new_features = torch.cat(feats_ls, dim=-1)
    return new_features

def pillar_pyramidV2(features, k=1, voxel_size=None, pc_range=None):
    """
    features shape [pillar_points, max_points, dim]
    points absolute coords: features[:,:,0]->x,  features[:,:,1]->y, features[:,:,2]->z
    """
    base_voxel = voxel_size #base pillar scale: 0.16
    scale_ls = []
    for i in range(k): #k=3
        scale_ls.append(base_voxel * 2**(i-1))
    scale_ls = sorted(scale_ls, reverse=True) #[0.32, 0.16, 0.08]
    feats_ls = []
    for scale in scale_ls:
        x_int = (features[:, :, 0].unsqueeze(2) / scale).round() 
        x_norm = x_int / (pc_range[3] / scale)
        x_res = features[:, :, 0].unsqueeze(2) - x_int * scale
        y_int = ((features[:, :, 1].unsqueeze(2) - pc_range[1]) / scale).round() 
        y_norm = y_int / ((pc_range[4] - pc_range[1]) / scale)
        y_res = (features[:, :, 1].unsqueeze(2) - pc_range[1]) - y_int * scale
        feats_ls.append(torch.cat([x_norm, x_res, y_norm, y_res], dim=-1)) #concat different scale info
    new_features = torch.cat([features, torch.cat(feats_ls, dim=-1)], dim=-1) #concat on pillar features
    return new_features

def pillar_pyramidV3(features, k=1, voxel_size=None, pc_range=None):
    """
    features shape [pillar_points, max_points, dim]
    points absolute coords: features[:,:,0]->x,  features[:,:,1]->y, features[:,:,2]->z
    """
    base_voxel = voxel_size #base pillar scale: 0.16
    scale_ls = []
    for i in range(k): #k=3
        scale_ls.append(base_voxel * 2**(i-1))
    scale_ls = sorted(scale_ls, reverse=True) #[0.32, 0.16, 0.08]
    feats_ls = []
    for scale in scale_ls:
        x_int = (features[:, :, 0].unsqueeze(2) / scale).round() 
        x_norm = x_int / (pc_range[3] / scale)
        x_res = features[:, :, 0].unsqueeze(2) - x_int * scale
        y_int = ((features[:, :, 1].unsqueeze(2) - pc_range[1]) / scale).round() 
        y_norm = y_int / ((pc_range[4] - pc_range[1]) / scale)
        y_res = (features[:, :, 1].unsqueeze(2) - pc_range[1]) - y_int * scale
        feats_ls.append(torch.cat([x_norm, x_res, y_norm, y_res], dim=-1)) #concat different scale info
    new_features = torch.cat([features, torch.cat(feats_ls, dim=-1)], dim=-1) #concat on pillar features
    return new_features

def HA_pillar(features, max_num, pc_range, actual_num, use_intensity=False):
    """
    features shape [pillar_points, max_points, dim]
    points absolute coords: features[:,:,0]->x,  features[:,:,1]->y, features[:,:,2]->z
    """
    max_value, min_value = features[:, :, 2].max(), features[:, :, 2].min()
    scale = (max_value - min_value) / max_num
    coord = torch.arange(0, max_num).to(features.device) + (min_value / scale).round()
    features[:, :, 2][features[:, :, 2]== 0.0] = max_value + scale
    h_int=(features[:, :, 2] / scale).floor().unsqueeze(-1)
    assert coord.shape[0] == max_num
    coord = coord.view(1,1,-1)
    h_hist =(h_int==coord).float().sum(-2)
    if use_intensity:
        feat_intensity = features[:, :, 3].sum(-1) / actual_num
        feat_intensity = feat_intensity.unsqueeze(-1)
        h_hist =  torch.cat((h_hist, feat_intensity), dim=1)
    # for i in range(features.shape[0]):
    #     # height_count[i] = torch.histc(features[i, :, 2][:int(actual_num[i])], bins=max_num, min=features[i, :, 2][:int(actual_num[i])].min(), max=features[i, :, 2][:int(actual_num[i])].max())
    #     his_z = torch.histc(features[i, :, 2][:int(actual_num[i])], bins=max_num, min=features[:, :, 2].min(), max=features[:, :, 2].max()) / max_num
    #     #avg_intensity = torch.mean(features[i, :, 3][:int(actual_num[i])]).unsqueeze(0)
    #     if use_intensity:
    #         height_count[i] = torch.cat((his_z, avg_intensity)).to(features.device)
    #     else:
    #         height_count[i] = his_z
    # height_count = torch.stack(height_count)
    return h_hist

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer_womax(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        return x

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM #True
        self.with_distance = self.model_cfg.WITH_DISTANCE #False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ #True
        num_point_features += 6 if self.use_absolute_xyz else 3 #4+6
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) #10->64

        self.voxel_x = voxel_size[0] #0.16
        self.voxel_y = voxel_size[1] #0.16
        self.voxel_z = voxel_size[2] #4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #[0, -39.68, -3, 69.12, 39.68, 1]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1) #actual_num:每个pillar内的实际点数
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1 #[1, -1]
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #voxel_features: 每个pillar的[25993, 32, 4] pillar个数，每个pillar内最大点数32，每个点的坐标xyz (原始点云的xyz坐标), intensity.
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] 
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) #每个pillar内的点的xyz分别求均值 [25993, 1, 3] 
        #25993个pillar 每个pillar内的点的x/y/z各自的均值
        f_cluster = voxel_features[:, :, :3] - points_mean #[25993, 32, 3] 每个pillar内部的原始点减去x,y,z的均值
        
        #每个pillar 原始x/y/z减去对应的pillar的中心点的x/y/z坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset) #pillar整数坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) #original: xyzitensity, pillar内点的xyz均值，以及每个点相对于pillar中心点的xyz offset差值。

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class PyramidPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM #True
        self.with_distance = self.model_cfg.WITH_DISTANCE #False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ #True
        self.bits = self.model_cfg.BITS
        num_point_features += 6 if self.use_absolute_xyz else 3 #4+6
        if self.with_distance:
            num_point_features += 1
        num_point_features += num_point_features*self.bits
        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) #10->64

        self.voxel_x = voxel_size[0] #0.16
        self.voxel_y = voxel_size[1] #0.16
        self.voxel_z = voxel_size[2] #4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #[0, -39.68, -3, 69.12, 39.68, 1]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #voxel_features: 每个pillar的[25993, 32, 4] pillar个数，每个pillar内最大点数32，每个点的坐标xyz (原始点云的xyz坐标), intensity.
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] 
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) #每个pillar内的点的xyz分别求均值 [25993, 1, 3] 
        #25993个pillar 每个pillar内的点的x/y/z各自的均值
        f_cluster = voxel_features[:, :, :3] - points_mean #[25993, 32, 3] 每个pillar内部的原始点减去x,y,z的均值
        
        #每个pillar 原始x/y/z减去对应的pillar的中心点的x/y/z坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset) #pillar整数坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) #original: xyzitensity, pillar内点的xyz均值，以及每个点相对于pillar中心点的xyz offset差值。
        features = pillar_pyramid(features, k=self.bits)
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class PyramidPillarVFEV2(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM #True
        self.with_distance = self.model_cfg.WITH_DISTANCE #False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ #True
        self.bits = self.model_cfg.BITS
        self.base_voxel_size = voxel_size[0]
        self.pc_range = np.array(self.model_cfg.PC_RANGE)
        num_point_features += 6 if self.use_absolute_xyz else 3 #4+6
        if self.with_distance:
            num_point_features += 1
        num_point_features += 4 * self.bits
        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) #10->64

        self.voxel_x = voxel_size[0] #0.16
        self.voxel_y = voxel_size[1] #0.16
        self.voxel_z = voxel_size[2] #4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #[0, -39.68, -3, 69.12, 39.68, 1]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #voxel_features: 每个pillar的[25993, 32, 4] pillar个数，每个pillar内最大点数32，每个点的坐标xyz (原始点云的xyz坐标), intensity.
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] 
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) #每个pillar内的点的xyz分别求均值 [25993, 1, 3] 
        #25993个pillar 每个pillar内的点的x/y/z各自的均值
        f_cluster = voxel_features[:, :, :3] - points_mean #[25993, 32, 3] 每个pillar内部的原始点减去x,y,z的均值
        
        #每个pillar 原始x/y/z减去对应的pillar的中心点的x/y/z坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset) #pillar整数坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) #original: xyzitensity, pillar内点的xyz均值，以及每个点相对于pillar中心点的xyz offset差值。
        features = pillar_pyramidV2(features, k=self.bits, voxel_size=self.base_voxel_size, pc_range=self.pc_range)
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class HA_PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM #True
        self.use_intensity = self.model_cfg.USE_INTENSITY
        self.with_distance = self.model_cfg.WITH_DISTANCE #False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ #True
        self.base_voxel_size = voxel_size[0]
        self.pc_range = np.array(self.model_cfg.PC_RANGE)
        num_point_features += 6 if self.use_absolute_xyz else 3 #4+6
        if self.with_distance:
            num_point_features += 1
        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        if self.use_intensity:
            num_filters = [32+1] + list(self.num_filters)
        else:
            num_filters = [32] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_womax(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) #10->64

        self.voxel_x = voxel_size[0] #0.16
        self.voxel_y = voxel_size[1] #0.16
        self.voxel_z = voxel_size[2] #4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #[0, -39.68, -3, 69.12, 39.68, 1]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #voxel_features: 每个pillar的[25993, 32, 4] pillar个数，每个pillar内最大点数32，每个点的坐标xyz (原始点云的xyz坐标), intensity.
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] 
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) #每个pillar内的点的xyz分别求均值 [25993, 1, 3] 
        #25993个pillar 每个pillar内的点的x/y/z各自的均值
        f_cluster = voxel_features[:, :, :3] - points_mean #[25993, 32, 3] 每个pillar内部的原始点减去x,y,z的均值
        
        # 每个pillar 原始x/y/z减去对应的pillar的中心点的x/y/z坐标
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset) #pillar整数坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) #original: xyzitensity, pillar内点的xyz均值，以及每个点相对于pillar中心点的xyz offset差值。
        voxel_count = features.shape[1]
        features = HA_pillar(features, voxel_count, self.pc_range, voxel_num_points, self.use_intensity)
        # mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        # features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict