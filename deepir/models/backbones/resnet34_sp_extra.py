import torch
import torch.nn as nn

from mmdet.models import BACKBONES

from .resnet_sp import ResNetSP

@BACKBONES.register_module
class ResNet34SP_ExtraLayers(ResNetSP):

    # Extra layers setting is determined by input size
    extra_setting = {
        300: {'outplanes': (256, 512, 128, 256, 128, 256, 128, 256),
              'kernel_size': (1, 3 , 1, 3, 1, 3, 1, 3),
              'stride': (1, 2, 1, 2, 1, 2, 1, 2),
              'padding': (0, 1, 0, 1, 0, 1, 0, 1)},
        512: {'outplanes': (256, 512, 128, 256, 128, 256, 128, 256, 128, 256),
              'kernel_size': (1, 3 , 1, 3, 1, 3, 1, 3, 1, 4),
              'stride': (1, 2, 1, 2, 1, 2, 1, 2, 1, 1),
              'padding': (0, 1, 0, 1, 0, 1, 0, 1, 0, 1)}
    }

    # The outplanes number of res layers
    res_layers_outplanes = (64, 128, 256, 512)

    def __init__(self,
                 input_size,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 use_sp_attn_indices=(0, 1, 2, 3)):
        
        super(ResNet34SP_ExtraLayers, self).__init__(
            depth=34,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            plugins=plugins,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            pretrained=pretrained,
            init_cfg=init_cfg,
            use_sp_attn_indices=use_sp_attn_indices)
        
        assert input_size in (300, 512), f"{input_size} is unsupported input size of images"
        self.input_size = input_size

        self.extra_layers_inplane = self.res_layers_outplanes[out_indices[-1]]
        self.extra_layers = self._make_extra_layers(self.extra_setting[input_size])

    def forward(self, x):
        outs, sp_attns = super().forward(x)
        # The forward of super returns tuple object
        # So they need to be converted to list object
        x = outs[-1]

        outs_extra = []
        for i, layer in enumerate(self.extra_layers):
            x = layer(x)
            if i % 2 == 1:
                outs_extra.append(x)
        
        return outs, tuple(outs_extra), sp_attns

    def _make_extra_layers(self, extra_setting):
        outplanes = extra_setting['outplanes']
        kernel_size = extra_setting['kernel_size']
        stride = extra_setting['stride']
        padding = extra_setting['padding']
        extra_layers = []
        for i in range(len(outplanes)):
            if i == 0:
                inplane = self.extra_layers_inplane
            else:
                inplane = outplanes[i-1]
            extra_layers.append(nn.Sequential(
                nn.Conv2d(inplane, outplanes[i], kernel_size[i], stride[i], padding[i]),
                nn.ReLU()
            ))

        return nn.Sequential(*extra_layers)
