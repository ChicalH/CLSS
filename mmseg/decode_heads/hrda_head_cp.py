
from copy import deepcopy

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead
import pdb

def scale_box(box, scale):
    y1, y2, x1, x2 = box
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@HEADS.register_module()
class HRDAHead_cp(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        elif single_scale_head == 'CPHead_for_hrda':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead_cp, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = 'DAFormerHead' 
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs,label):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            preds1,preds2 = self.head(features,label)

            crop_seg_logits = preds1['seg']
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            preds1['seg'] = preds / count_mat

            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            crop_seg_logits = preds2['seg']
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            preds2['seg']= preds / count_mat

            return preds1,preds2
        else:
            return self.head(inp,label)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs,label=None,hr_cropped_seg_label=None):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box
        lr_seg1,lr_seg2_ = self.head(lr_inp,label)
        hr_seg1,hr_seg2_ = self.decode_hr(hr_inp, batch_size,hr_cropped_seg_label)

        att = self.get_scale_attention(lr_sc_att_inp)

        lr_seg2 = lr_seg2_['seg']
        if has_crop:
            mask = lr_seg2.new_zeros([lr_seg2.shape[0], 1, *lr_seg2.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask

        lr_seg2 = (1 - att) * lr_seg2
        up_lr_seg2 = self.resize(lr_seg2, hr_scale / lr_scale)

        lr_seg = lr_seg1['seg']

        lr_seg = (1 - att) * lr_seg
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)

        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        hr_seg2 = hr_seg2_['seg']
        if has_crop:
            
            hr_seg2_inserted = torch.zeros_like(up_lr_seg2)
            slc = self.hr_crop_slice(self.os)
            hr_seg2_inserted[:, :, slc[0], slc[1]] = hr_seg2
        else:
            hr_seg2_inserted = hr_seg2

        fused_seg2 = att * hr_seg2_inserted + up_lr_seg2

        hr_seg = hr_seg1['seg']
        if has_crop:
            
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg


        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return [fused_seg, lr_seg, hr_seg],[lr_seg1,lr_seg2_],[hr_seg1,hr_seg2_],[fused_seg2, lr_seg2, hr_seg2]
    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
            hr_cropped_seg_label = crop(gt_semantic_seg, self.hr_crop_box)
            self.debug_output['Cropped GT'] = \
                hr_cropped_seg_label.squeeze(1).detach().cpu().numpy()
        else:
            hr_cropped_seg_label = gt_semantic_seg

        seg_logits = self.forward(inputs,gt_semantic_seg,hr_cropped_seg_label)
        losses = self.losses(seg_logits, gt_semantic_seg,hr_cropped_seg_label, seg_weight)

        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs,None,None)[0][0]

    def losses(self, seg, seg_label,hr_cropped_seg_label, seg_weight=None):
        """Compute losses."""
        seg_logit,lr_seg_cp,hr_seg_cp,seg_logit2 = seg

        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead_cp, self).losses(fused_seg, seg_label, seg_weight)

        fused_seg2, lr_seg2, hr_seg2 = seg_logit2
        loss.update(add_prefix(super(HRDAHead_cp, self).losses(fused_seg2, seg_label,seg_weight), 'seg2'))
        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(lr_seg2, seg_label,
                                                 seg_weight), 'lr_seg2'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:

            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight

            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(hr_seg2, hr_cropped_seg_label,
                                                 cropped_seg_weight), 'hr_seg2'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(hr_seg2, seg_label,
                                                 seg_weight), 'hr_seg2'))

        seg2_weight = 0.7
        loss['seg2.loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)*seg2_weight
        if self.lr_loss_weight > 0:
            loss['lr_seg2.loss_seg'] *= self.lr_loss_weight*seg2_weight
        if self.hr_loss_weight > 0:
            loss['hr_seg2.loss_seg'] *= self.hr_loss_weight*seg2_weight
 
        loss_cp_lr = self.head.losses(lr_seg_cp, seg_label, seg_weight)
        loss_cp_hr = self.head.losses(hr_seg_cp, hr_cropped_seg_label, seg_weight)
        loss.update(add_prefix(loss_cp_lr,'loss_cp_lr'))
        loss.update(add_prefix(loss_cp_hr,'loss_cp_hr'))


        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:

            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight

            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(hr_seg, hr_cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead_cp, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))


        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        return loss
