import torch
from torch import nn
from torchvision import transforms

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

from ..utils import tensor_to_img, heatmap_over_img, mask_top_rate, denormalize

@DETECTORS.register_module()
class FasterRCNN_AAL(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 ### AAL MODIFIED ###
                 fgsm_epsilon=0.01,
                 back_rate = 0.01,
                 ### END MODIFIED ###
                 ### VISUAL MODIFIED ###
                 visualization = False,
                 visual_dir = None
                 ### END MODIFIED ###
                 ):
        super(FasterRCNN_AAL, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        ### AAL MODIFIED ###
        self.fgsm_epsilon = fgsm_epsilon
        self.back_rate = back_rate
        self.mask_pool_layer = nn.AvgPool2d(kernel_size=15,stride=1,padding=7)
        ### END MODIFIED ###
        ### VISUAL MODIFIED ###
        self.visualization = visualization,
        self.visual_dir = visual_dir
        self.forward_count = 1
        ### END MODIFIED ###
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        ### AAL MODIFIED ###
        x, sp_attn = self.backbone(img)
        ### END MODIFIED ###
        if self.with_neck:
            x = self.neck(x)
        ### AAL MODIFIED ###
        return x, sp_attn
        ### END MODIFIED ###
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        ### AAL MODIFIED ###
        x, sp_attn = self.extract_feat(img)
        ### END MODIFIED ###
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ### AAL MODIFIED ###
        adv_delta = torch.zeros_like(img).to(img.device).requires_grad_()
        x, sp_attns = self.extract_feat(img+adv_delta)

        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        loss_sum = torch.zeros([1]).to(losses['loss_rpn_cls'][0].device)
        for i in range(len(losses['loss_rpn_cls'])):
            loss_sum += losses['loss_rpn_cls'][i]
        for i in range(len(losses['loss_rpn_bbox'])):
            loss_sum += losses['loss_rpn_bbox'][i]
        loss_sum += losses['loss_cls']
        loss_sum += losses['acc']
        loss_sum += losses['loss_bbox']

        loss_sum.backward()
        adv_delta_grad = adv_delta.grad
        max_across_channels, _ = torch.max(adv_delta_grad, dim=1, keepdim=True)
        back_mask = mask_top_rate(max_across_channels, self.back_rate)
        one = torch.ones_like(back_mask)
        sp_attn_stem_resized = transforms.Resize((img.shape[2],img.shape[3]))(sp_attns[-1])
        backtracked_sp_attn_stem = (one - self.mask_pool_layer(back_mask)) * sp_attn_stem_resized
        adv_delta = (adv_delta + backtracked_sp_attn_stem * self.fgsm_epsilon * torch.sign(adv_delta_grad)).detach()

        ### VISUAL MODIFIED ###
        if self.visualization and self.forward_count % 500 == 0:
            adv_delta_grad_img = tensor_to_img((adv_delta_grad[0]-torch.min(adv_delta_grad[0]))/(torch.max(adv_delta_grad[0])-torch.min(adv_delta_grad[0])))
            back_mask_img = tensor_to_img(back_mask[0])
            sp_attn_stem_resized_img = tensor_to_img(sp_attn_stem_resized[0])
            backtracked_sp_attn_stem_img = tensor_to_img(backtracked_sp_attn_stem[0])
            img_img = tensor_to_img(denormalize(img[0]))
            heatmap = heatmap_over_img(denormalize(img[0]), sp_attn_stem_resized[0])
            heatmap_backtracked = heatmap_over_img(denormalize(img[0]), backtracked_sp_attn_stem[0])
            img_attacked = tensor_to_img((denormalize(img)+adv_delta)[0])
            adv_delta_img = tensor_to_img(adv_delta[0]/self.fgsm_epsilon)
            adv_delta_grad_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_adv_delta_grad.png')
            back_mask_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_back_mask.png')
            sp_attn_stem_resized_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_sp_attn.png')
            backtracked_sp_attn_stem_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_back_sp_attn.png')
            img_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_img.png')
            heatmap.save(self.visual_dir+'/'+str(self.forward_count//500)+'_heatmap.png')
            heatmap_backtracked.save(self.visual_dir+'/'+str(self.forward_count//500)+'_heatmap_back.png')
            img_attacked.save(self.visual_dir+'/'+str(self.forward_count//500)+'_img_attacked.png')
            adv_delta_img.save(self.visual_dir+'/'+str(self.forward_count//500)+'_adv_delta.png')
        self.forward_count += 1
        ### END MODIFIED ###


        x, sp_attns = self.extract_feat(img+adv_delta)
        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        ### END MODIFIED ###

        return losses
    
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        ### AAL MODIFIED ###
        x, sp_attn = self.extract_feat(img)
        ### END MODIFIED ###

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        ### AAL MODIFIED ###
        x, sp_attn = self.extract_feat(img)
        ### END MODIFIED ###
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        ### AAL MODIFIED ###
        x, sp_attns = self.extract_feats(imgs)
        ### END MODIFIED ###
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        ### AAL MODIFIED ###
        x, sp_attn = self.extract_feat(img)
        ### END MODIFIED ###
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
