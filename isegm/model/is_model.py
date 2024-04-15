import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ops import BatchImageNormalize, ScaleLayer, DistMaps
from .modifiers import LRMult


class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, clicks_groups=None,
                 with_prev_mask=False, use_leaky_relu=False, binary_prev_mask=False,
                 conv_extend=False, norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(3 + self.coord_feature_ch, 6 + self.coord_feature_ch, kernel_size=1),
                nn.BatchNorm2d(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(6 + self.coord_feature_ch, 3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(self.coord_feature_ch, 64, 3, 2, 1)
            self.maps_transform.apply(LRMult(0.1))  # Adjusting Learning Rate
        else:
            self.rgb_conv = None
            self.maps_transform = nn.Sequential(
                nn.Conv2d(self.coord_feature_ch, 16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(16, 64, 3, 2, 1),
                ScaleLayer(init_value=0.05, lr_mult=1)  # Adjusting Learning Rate
            )

        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, use_disks=use_disks)

    def forward(self, image, points):
        # 이미지로부터 마스크 추출
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            outputs = self.backbone_forward(x)
        else:
            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features)

        outputs['instances'] = F.interpolate(outputs['instances'],
                                             size=image.size()[2:], mode='bilinear', align_corners=True)

        if self.with_aux_output:
            outputs['instances_aux'] = F.interpolate(outputs['instances_aux'],
                                                     size=image.size()[2:], mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            # image의 앞 3개 채널에 mask 정보 담겨있음
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    # def get_coord_features(self, image, prev_mask, points):
    #     if self.clicks_groups is not None:
    #         points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
    #         coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
    #         coord_features = torch.cat(coord_features, dim=1)
    #     else:
    #         coord_features = self.dist_maps(image, points)

    #     if prev_mask is not None:
    #         if coord_features is None:
    #             coord_features = torch.zeros_like(image)
    #         coord_features = torch.cat((prev_mask, coord_features), dim=1)

    #     return coord_features
    def get_coord_features(self, image, prev_mask, points):
        coord_features = torch.zeros((image.size(0), self.coord_feature_ch,
                                     image.size(2), image.size(3)), device=image.device)
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            dist_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            dist_features = torch.cat(dist_features, dim=1)
            coord_features[:, :dist_features.size(1), :, :] = dist_features

        if prev_mask is not None:
            coord_features[:, -1, :, :] = prev_mask.squeeze(1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device) for x in group_points]

    return group_points
