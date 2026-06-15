
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .daformer_head import DAFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPModule
from ..losses.loss import PixelPrototypeCELoss
from timm.models.layers import trunc_normal_
import pdb
from einops import rearrange, repeat
from .daformer_head import  DAFormerHead_shareproto,l2_normalize
import torch.nn.functional as F
import torch.distributed as dist
from ..losses import accuracy
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import numpy as np


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class CPHead_for_hrda(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(CPHead_for_hrda, self).__init__(input_transform='multiple_select', **kwargs)
        embedding_dim = self.channels 
        self.num_prototype = 15
        self.gamma = 0.999
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, embedding_dim),###[k,m,d]
                                       requires_grad=True)
        self.prototypes_parameter = nn.Parameter(torch.zeros(self.num_prototype*self.num_classes, self.num_classes),###[mk,k]
                                       requires_grad=True)
        trunc_normal_(self.prototypes, std=0.02)
        trunc_normal_(self.prototypes_parameter, std=0.02)
        self.mask_norm = nn.LayerNorm(kwargs['num_classes'])
        self.net_decoder1 = DAFormerHead_shareproto(**kwargs)
        self.net_decoder2 = DAFormerHead_shareproto(use_inter=True,**kwargs)
        self.loss_decode = PixelPrototypeCELoss(ignore_index=255)
 
    def consistent_embedding_representation(self, _c, _c2, out_seg,out_seg2, gt_seg, masks,masks2):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        proto_logits = cosine_similarity
        pred_seg2 = torch.max(out_seg2, 1)[1]
        mask2 = (gt_seg == pred_seg2.view(-1))
        cosine_similarity2 = torch.mm(_c2, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        proto_logits2 = cosine_similarity2
        proto_target = gt_seg.clone().float()
        proto_target2 = gt_seg.clone().float()
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            init_q2 = masks2[..., k]
            init_q2 = init_q2[gt_seg == k, ...]
            if init_q.shape[0] == 0  or init_q2.shape[0] == 0:
                continue
            sinkhorn_init_q = torch.cat((init_q,init_q2),dim=0)
            q, indexs,q2, indexs2 = consistent_ot(sinkhorn_init_q, init_q.shape[0] )
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile
            f = m_q.transpose(0, 1) @ c_q
            n = torch.sum(m_q, dim=0)
            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
            m_k2 = mask2[gt_seg == k]
            c_k2 = _c2[gt_seg == k, ...]
            m_k_tile2 = repeat(m_k2, 'n -> n tile', tile=self.num_prototype)
            m_q2 = q2 * m_k_tile2
            c_k_tile2 = repeat(m_k2, 'n -> n tile', tile=c_k2.shape[-1])
            c_q2 = c_k2 * c_k_tile2
            f2 = m_q2.transpose(0, 1) @ c_q2
            n2 = torch.sum(m_q2, dim=0)
            if torch.sum(n2) > 0:
                f2 = F.normalize(f2, p=2, dim=-1)
                new_value2 = momentum_update(old_value=protos[k, n2 != 0, :], new_value=f2[n2 != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n2 != 0, :] = new_value2
            proto_target2[gt_seg == k] = indexs2.float() + (self.num_prototype * k)
        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)


        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target,proto_logits2, proto_target2

    def forward(self, conv_out,label):
        _c1,shape=self.net_decoder1(conv_out)
        _c2,_=self.net_decoder2(conv_out)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        masks = torch.einsum('nd,kmd->nmk', _c1, self.prototypes)
        out_seg = torch.mm(masks.view(masks.shape[0],-1), self.prototypes_parameter)
        out_seg = self.mask_norm(out_seg)
        seg_nk = nn.functional.softmax(out_seg, dim=1)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=shape[0], h=shape[2])
        masks2 = torch.einsum('nd,kmd->nmk', _c2, self.prototypes)
        out_seg2 = torch.mm(masks2.view(masks2.shape[0],-1), self.prototypes_parameter)
        out_seg2 = self.mask_norm(out_seg2)
        seg_nk2 = nn.functional.softmax(out_seg2, dim=1)
        out_seg2 = rearrange(out_seg2, "(b h w) k -> b k h w", b=shape[0], h=shape[2])
        if label is not None:
            label = label.float()
            gt_seg = F.interpolate(label, size=(shape[2],shape[3]), mode='nearest').view(-1)
            contrast_logits, contrast_target,contrast_logits2, contrast_target2 = self.consistent_embedding_representation(_c1, _c2, out_seg, out_seg2,gt_seg, masks,masks2)
            seg = {'seg': out_seg, 'seg_nk': seg_nk, 'logits': contrast_logits, 'target': contrast_target}
            seg2 = {'seg': out_seg2, 'seg_nk': seg_nk2, 'logits': contrast_logits2, 'target': contrast_target2}
        else: 
            seg = {'seg': out_seg}
            seg2 = {'seg': out_seg2}
        return seg,seg2

    def forward_train(self,
                    inputs,
                    img_metas,
                    gt_semantic_seg,
                    train_cfg,
                    seg_weight=None,
                    return_logits=False):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs,gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits[0]['seg']
        return losses 

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs,None)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg, seg_label, seg_weight=None):
        loss = dict()
        seg1,seg2 = seg
        seg1['seg'] = resize(
            input=seg1['seg'],
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg2['seg'] = resize(
            input=seg2['seg'],
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        prototype_con_loss1 = self.loss_decode(seg1,seg_label,weight=seg_weight)
        prototype_con_loss2 = self.loss_decode(seg2,seg_label,weight=seg_weight)
        CRL_loss = torch.mean(torch.cosine_similarity(seg1['seg_nk'].float(), seg2['seg_nk'].float(),dim=1))
        loss['loss_seg'] = prototype_con_loss1 + prototype_con_loss2 - 0.01*CRL_loss
        loss['acc_seg'] = accuracy(seg1['seg'], seg_label)
        return loss

def consistent_ot(out, n1, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    B = L.shape[1]
    K = L.shape[0]
    sum_L = torch.sum(L)
    L /= sum_L
    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K
        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B
    L *= B
    L = L.t()
    indexs = torch.argmax(L, dim=1)
    L = F.gumbel_softmax(L, tau=0.5, hard=True)
    return L[:n1], indexs[:n1],L[n1:], indexs[n1:]
    
def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update
    

def softargmax2d(input, beta=100):
    b,c, h, w = input.shape
    input = nn.functional.softmax(beta * input, dim=1)
    indices = np.linspace(0, 1, c)
    indices = torch.tensor(indices).to(input.device)
    indices = indices.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    result = torch.sum((c - 1) * input * indices, dim=1)
    return result