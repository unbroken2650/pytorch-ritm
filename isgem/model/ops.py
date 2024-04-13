import torch
from torch import nn as nn
import numpy as np

# ritm의 핵심 method인 Distance Transform 수행


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.norm_radius = norm_radius
        self.spatial_scale = spatial_scale
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks

    # 각 점으로부터 이미지 내의 모든 픽셀까지의 거리 계산
    def get_coord_features(self, points, batch_size, rows, cols):

        # 이미지를 2d array로 변환
        num_points = points.shape[1] // 2
        points = points.view(-1, points.size(2))
        points, points.order = torch.split(points, [2, 1], dim=1)

        invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
        row_array = torch.arrange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arrange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
        coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

        # 거리 계산
        add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
        coords.add_(-add_xy)

        if not self.use_disks:
            coords.div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)

        coords[:, 0] += coords[:, 1]
        coords = coords[:, :1]

        coords[invalid_points, :, :, :] = 1e6

        coords = coords.view(-1, num_points, 1, rows, cols)
        coords = coords.min(dim=1)[0]
        coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(torch.full((1,), init_value / lr_mult, dtype=torch.float32))

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x.scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor.sub_(self.mean.to(tensor.device).div_(self.std.to(self.device)))
        return tensor
