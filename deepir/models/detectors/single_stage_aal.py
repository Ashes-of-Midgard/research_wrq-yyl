# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from torchvision import transforms

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector

from ..dense_heads import SSDHeadAAL, SSDHeadSA


@DETECTORS.register_module()
class SingleStageDetectorAAL(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 #AAL args
                 fgsm_epsilon=0.01,
                 back_rate = 0.01):
        super(SingleStageDetectorAAL, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        ### AAL MODIFIED ###
        self.fgsm_epsilon = fgsm_epsilon
        self.back_rate = back_rate
        ### END MODIFIED ###

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        ### AAL MODIFIED ###
        feats, feats_extra, sp_attn = self.backbone(img)
        if self.with_neck:
            feats, feats_extra, sp_attn = self.neck(feats, feats_extra, sp_attn)
        return feats, feats_extra, sp_attn
        ### END MODIFIED ###

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        ### AAL MODIFIED ###
        feats, feats_extra, _ = self.extract_feat(img)
        feats = list(feats)
        feats.extend(feats_extra)
        ### END MODIFIED ###
        outs = self.bbox_head(feats)
        
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetectorAAL, self).forward_train(img, img_metas)
        ### AAL MODIFIED ###
        adv_delta = torch.zeros_like(img).to(img.device).requires_grad_()
        # 1. First forward: clean inference
        feats, feats_extra, sp_attns = self.extract_feat(img+adv_delta)
        feats = list(feats)
        feats.extend(feats_extra)
        losses = self.bbox_head.forward_train(feats, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        loss_sum = torch.zeros([1]).to(losses['loss_cls'][0].device)
        for i in range(len(losses['loss_cls'])):
            loss_sum += losses['loss_cls'][i]
        for i in range(len(losses['loss_bbox'])):
            loss_sum += losses['loss_bbox'][i]
        # 2. Backtrack: Adjust the spatial attention
        loss_sum.backward()
        adv_delta_grad = adv_delta.grad
        max_across_channels, _ = torch.max(adv_delta_grad, dim=1, keepdim=True)
        back_mask = mask_top_rate(max_across_channels, self.back_rate)
        one = torch.ones_like(back_mask)
        sp_attn_stem_resized = transforms.Resize((img.shape[2],img.shape[3]))(sp_attns[-1])
        backtracked_sp_attn_stem = (((one - 0.05 * back_mask) * sp_attn_stem_resized).detach())
        #for i in range(len(sp_attns)):
        #    if sp_attns[i] is not None:
        #        back_mask_resized = transforms.Resize((sp_attns[i].shape[2],sp_attns[i].shape[3]))(back_mask)
        #        print(f'back_mask_resized: {back_mask_resized.shape}')
        #        one = torch.ones_like(back_mask_resized)
        #        backtracked_sp_attns.append(((one - 0.05 * back_mask_resized) * sp_attns[i]).detach())
        #    else:
        #        backtracked_sp_attns.append(None)
        # 3. Assign backtracked sp attns to the backbones' layers
        # 4. Second forward: Select cricial region for adversarial attacks
        adv_delta = (backtracked_sp_attn_stem * self.fgsm_epsilon * torch.sign(adv_delta_grad)).detach()
        feats, feats_extra, sp_attns = self.extract_feat(img+adv_delta)
        feats = list(feats)
        feats.extend(feats_extra)
        losses = self.bbox_head.forward_train(feats, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        #feats, feats_extra, sp_attns = self.extract_feat(img)
        #if type(self.bbox_head) in (SSDHeadSA, SSDHeadAAL):
        #    losses = self.bbox_head.forward_train(feats, feats_extra, sp_attns, img_metas, gt_bboxes,
        #                                          gt_labels, gt_bboxes_ignore)
        #else:
        #    # if the head is SSDHead, use this branch
        #    feats = list(feats)
        #    feats.extend(feats_extra)
        #    losses = self.bbox_head.forward_train(feats, img_metas, gt_bboxes,
        #                                          gt_labels, gt_bboxes_ignore)
        ### END MODIFIED ###
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        ### AAL MODIFIED ###
        feats, feats_extra, _ = self.extract_feat(img)
        feats = list(feats)
        feats.extend(feats_extra)
        ### END MODIFIED ###
        results_list = self.bbox_head.simple_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        ### AAL MODIFIED ###
        feats, feats_extra, _ = self.extract_feat(imgs)
        feats = list(feats)
        feats.extend(feats_extra)
        ### END MODIFIED ###
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        ### AAL MODIFIED ###
        feats, feats_extra, _ = self.extract_feat(img)
        feats = list(feats)
        feats.extend(feats_extra)
        ### END MODIFIED ###
        outs = self.bbox_head(feats)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
    

def mask_top_rate(data, kept_rate):
    """
    Given a batch of samples(in a tensor form) and a rate of kept values,
    return a mask tensor which has the same shape as the input tensor. And
    the positions where the input tensor's value is within its topk range
    determined by the kept rate are set to be 1, others are set to be 0.

    Args:
        data (Tensor): The input samples
        kept_rate (float): The kept rate. Of each sample in the batch, how
            much ratio of the values from top are kept.

    Returns:
        Tensor: The mask tensor in the same shape as the input tensor.
    """
    data_flattened = data.view(data.size(0), -1)
    values_kept, _ = data_flattened.topk(int(data_flattened.size(1)*kept_rate), dim=1)
    values_min, _ = torch.min(values_kept, dim=-1)
    values_min = values_min.unsqueeze(-1).repeat(1, data_flattened.size(-1))
    mask = torch.ge(data_flattened, values_min).float().view(data.size())
    return mask