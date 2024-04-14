import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ocr import SpatialOCR_Module, SpatialGather_Module
from .resnetv1b import BasicBlockV1b, BottleNeckV1b

logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.align_corners = align_corners
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        # branches
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index]*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index]*block.expansion)
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                      num_channels[branch_index], stride=stride, downsample=downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:  # High res. -> Low res.
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, bias=False),
                            nn.BatchNorm2d(num_inchannels[i])
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)
                                ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet(nn.Module):

    def __init__(self, width, num_classes, ocr_width=256, small=False, align_corners=True):
        super(HighResolutionNet, self).__init__()
        self.width = width
        self.ocr_width = ocr_width
        self.align_corners = align_corners

        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        num_blocks = 2 if small else 4

        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleNeckV1b, 64, stage1_num_channels, num_blocks=num_blocks)
        stage1_out_channel = BottleNeckV1b.expansion * stage1_num_channels

        self.stage2_num_branches = 2
        num_channels = [width, 2*width]
        num_inchannels = [num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1,
            num_branches=self.stage2_num_branches, num_blocks=2*[num_blocks], num_channels=num_channels)

        self.stage3_num_branches = 3
        num_channels = [width, 2*width, 2*2*width]
        num_inchannels = [num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=3 if small else 4,
            num_branches=self.stage3_num_branches, num_blocks=3*[num_blocks], num_channels=num_channels)

        self.stage4_num_branches = 4
        num_channels = [width, 2*width, 2*2*width, 2*2*2*width]
        num_inchannels = [num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3,
            num_branches=self.stage4_num_branches, num_blocks=4*[num_blocks], num_channels=num_channels)

        last_inp_channels = int(np.sum(pre_stage_channels))

        if self.ocr_width > 0:
            ocr_mid_channels = 2 * self.ocr_width
            ocr_key_channels = self.ocr_width

            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(last_inp_channels, ocr_mid_channels, 3, 1, 1),
                nn.BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True)
            )

            self.ocr_gather_head = SpatialGather_Module(num_classes)
            self.ocr_distri_head = SpatialOCR_Module(
                ocr_mid_channels, ocr_key_channels, ocr_mid_channels,
                scale=1, dropout=0.05, norm_layer=nn.BatchNorm2d, align_corners=align_corners)
            self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, 1, 1, 0, bias=True)
            self.aux_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels, 1, 1, 0),
                nn.BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_inp_channels, num_classes, 1, 1, 0, bias=True)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels, 3, 1, 1),
                nn.BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_inp_channels, num_classes, 1, 1, 0, bias=True)
            )

    def _make_transition_layer(self, channels_pre_layer, channels_cur_layer):
        num_branches_pre = len(channels_pre_layer)
        num_branches_cur = len(channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # high Res. -> low Res.
                if channels_cur_layer[i] != channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(channels_pre_layer[i], channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = channels_pre_layer[-1]
                    outchannels = channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels, num_modules, num_branches, num_blocks, num_channels, fuse_method='SUM', multi_scale_output=True):
        modules = []

        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks,
                    num_inchannels, num_channels,
                    fuse_method, reset_multi_scale_output, align_corners=self.align_corners
                )
            )

            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, additional_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if additional_features is not None:
            x = x + additional_features

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)

        feats = torch.cat([x[0], x1, x2, x3], 1)

        if self.ocr_width > 0:
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)

            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)
            return [out, out_aux]
        else:
            return [self.cls_head(feats), None]


    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()

        if not os.path.exists(pretrained_path):
            print(f'\nFile "{pretrained_path}" does not exist.')
            print('You need to specify the correct path to the pre-trained weights.\n'
                'You can download the weights for HRNet from the repository:\n'
                'https://github.com/HRNet/HRNet-Image-Classification')
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                        pretrained_dict.items()}

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
