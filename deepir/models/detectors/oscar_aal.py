# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import numpy as np
import torch
from torchvision import transforms
import mmcv
from mmcv.runner import auto_fp16
from mmdet.core import bbox2result
from mmdet.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)
from mmdet.models.detectors import BaseDetector

from deepir.models.builder import build_heuristic

@DETECTORS.register_module()
class OSCARNet_AAL(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """
##############################################################################
################################# initialize #################################
##############################################################################

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
        super(OSCARNet_AAL, self).__init__(init_cfg)
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

##############################################################################
########################## shared by train and test ##########################
##############################################################################

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        ### AAL MODIFIED ###
        x, sp_attns = self.backbone(img)
        ### END MODIFIED ###
        if self.with_neck:
            x = self.neck(x)
        ### AAL MODIFIED ###
        return x, sp_attns
        ### END MODIFIED ###

##############################################################################
################################# train only #################################
##############################################################################

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_noco_map,
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
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        ### AAL MODIFIED ###
        adv_delta = torch.zeros_like(img).to(img.device).requires_grad_()
        
        x, sp_attns = self.extract_feat(img+adv_delta)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_noco_map,
                                              gt_bboxes_ignore)
        loss_sum = torch.zeros([1]).to(losses['loss_coarse_cls'].device)
        loss_sum += losses['loss_coarse_cls']
        loss_sum += losses['loss_coarse_bbox']
        loss_sum += losses['loss_refine_cls']
        loss_sum += losses['loss_refine_bbox']
        loss_sum += losses['loss_refine_noco']
        
        loss_sum.backward()
        adv_delta_grad = adv_delta.grad
        max_across_channels, _ = torch.max(adv_delta_grad, dim=1, keepdim=True)
        back_mask = mask_top_rate(max_across_channels, self.back_rate)
        one = torch.ones_like(back_mask)
        sp_attn_stem_resized = transforms.Resize((img.shape[2],img.shape[3]))(sp_attns[-1])
        backtracked_sp_attn_stem = (((one - 0.05 * back_mask) * sp_attn_stem_resized).detach())
        adv_delta = (backtracked_sp_attn_stem * self.fgsm_epsilon * torch.sign(adv_delta_grad)).detach()

        x, sp_attns = self.extract_feat(img+adv_delta)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_noco_map,
                                              gt_bboxes_ignore)
        ### END MODIFIED ###
        return losses

##############################################################################
################################## test only #################################
##############################################################################

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
        feat, _ = self.extract_feat(img)
        ### END MODIFIED ###
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

##############################################################################
################################### ignored ##################################
##############################################################################

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

        feats = self.extract_feats(imgs)
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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
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

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        ### AAL MODIFIED ###
        x, _ = self.extract_feat(img)
        ### END MODIFIED ###
        outs = self.bbox_head(x)
        return outs

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
