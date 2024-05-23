import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from .resnet34_sp import ResNetSPSSD, Bottleneck, BasicBlock, SpatialAttention
from mmcv.cnn import constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint
from mmcv.cnn import (ContextBlock, GeneralizedAttention, build_conv_layer,
                      build_norm_layer)
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import get_root_logger
from mmdet.models import BACKBONES
import pdb

@BACKBONES.register_module
class SSDR34SP(ResNetSPSSD):

    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 input_size,
                 depth,
                 num_stages,
                 out_indices,
                 frozen_stages,
                 norm_cfg,
                 style,
                 l2_norm_scale=20.):
        # TODO: in_channels for mmcv.VGG
        super(SSDR34SP, self).__init__(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            style=style
        )
        assert input_size in (300, 512)
        self.input_size = input_size


        self.inplanes = 512
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(
            256,
            # 1024,
            l2_norm_scale)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)


            # for m in self.features.modules():
            #     if isinstance(m, nn.Conv2d):
            #         kaiming_init(m)
            #     elif isinstance(m, nn.BatchNorm2d):
            #         constant_init(m, 1)
            #     elif isinstance(m, nn.Linear):
            #         normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.conv0(x)
        # sp= self.sp_layer(x)  #spatial attention
        # x = sp * x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # import pdb;pdb.set_trace()
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 0 or i ==1:
                x = res_layer(x)
            else:
                x,sp = res_layer(x)
            if i in self.out_indices:
                # pdb.set_trace()
                # sp= self.sp_layer(x)  #spatial attention
                # x = sp * x
            # if i == self.out_indices:
                outs.append(x)
        # pdb.set_trace()
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        # pdb.set_trace()
        outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0], sp
        else:
            return tuple(outs), sp

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
