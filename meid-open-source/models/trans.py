
# --------------------------------------------------------
# Trans
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li
# --------------------------------------------------------
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Attention
from timm.models.helpers import build_model_with_cfg
from .net_module import NeXtVLAD

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 157, 'input_size': (2048, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embeds.0.proj', 'classifier': 'head',
        **kwargs
    }
#157 for charades
#1004 for videolt

default_cfgs = {
    'aaa': _cfg(
        url='',
        ),
}

Size_ = Tuple[int, int]

class BGLayer(nn.Module):
    def __init__(self, channel=150, reduction=3): #3
        super(BGLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, c = x.size()
        y = self.avg_pool(x).view(b, n)
        y = self.fc(y).view(b, n, 1)

        return y.expand_as(x)


class AttnBlock(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x): #, size: Size_
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = self.sr(x)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


import copy
def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Mlp_timm(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_timm_expert4(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwitchFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer, drop,
                capacity_factor=1.0, drop_tokens=False, n_experts=2):
        super().__init__()

        self.expert1 = Mlp_timm(in_features=in_features, hidden_features=hidden_features, act_layer=act_layer, drop=drop)

    def forward(self, x, y=None):
        batch_size, seq_len, d_model = x.shape

        x1 = self.expert1(x)
        x2 = x1

        final_output = x1

        return final_output, [x1,x2]

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            self.attn = AttnBlock(dim, num_heads, attn_drop, drop, sr_ratio)
        else:
            self.attn = AttnBlock(dim, num_heads, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x): #, size: Size_
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_MoE_v8(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwitchFeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y=None): #, size: Size_

        x = x + self.drop_path(self.norm1(x))

        x1 = self.norm2(x)

        final_output, z2 = self.mlp(x1, y)

        final_output = x1 + self.drop_path(final_output)

        return final_output, z2, x1


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv1d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim), )
        self.stride = stride

    def forward(self, x): #, size: Size_
        B, N, C = x.shape

        cnn_feat_token = x.transpose(1, 2)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token

        x = x.transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:

        B, N, C = x.shape

        x = self.proj(x)
        x = self.norm(x)

        return x


class Trans_Pos(nn.Module):

    def __init__(
            self, img_size=224, patch_size=4, in_chans=2048, num_classes=1004, embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), wss=None,
            block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.bgl = BGLayer()

        for p in self.parameters():
            p.requires_grad = False       
             
        self.drop_path2 = DropPath(0.)

        self.expert4 = Mlp_timm_expert4(in_features=2048, hidden_features=2048*8, act_layer=nn.GELU, drop=0)

        self.pos_block2 = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm2 = nn.BatchNorm1d(self.num_features)

        self.drop_path_diff = DropPath(0.)
        self.expert_diff = Mlp_timm_expert4(in_features=2048, hidden_features=2048*8, act_layer=nn.GELU, drop=0)
        self.pos_block_diff = PosConv(embed_dims[0], embed_dims[0])
        self.norm_diff = nn.BatchNorm1d(self.num_features)
        self.blend = nn.Linear(300, 150)
        
        # classification head
        self.head2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x, x_diff, y=None, phase='val'):
        B = x.shape[0]
        z=[]
        output = None

        importance = self.bgl(x)
        x1 = x / (importance+0.5)

        for i, (embed, drop, blocks, pos_blk, pos_blk1) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block, self.pos_block2)):

            x = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x, z2, branch = blk(x, output)

                branch = x1
                
                final_output = self.expert4(branch)
                final_output_diff = self.expert_diff(x_diff)

                branch = branch + self.drop_path2(final_output)
                branch_diff = x_diff + self.drop_path_diff(final_output_diff)
                del final_output

                z.append(z2)
                if j == 0:
                    x = pos_blk(x)
                    branch = pos_blk1(branch)
                    branch_diff = self.pos_block_diff(branch_diff)

        del output
        x = self.norm(x)
        
        branch = self.norm2(branch.transpose(2,1)).transpose(1,2)
        branch_diff = self.norm_diff(branch_diff.transpose(2,1)).transpose(1,2)
        branch = torch.cat((branch, branch_diff),dim=1)
        branch = self.blend(branch.transpose(2,1)).transpose(1,2)

        x = x * importance
        return x.mean(dim=1), z, branch.mean(dim=1)  # GAP here

    def forward(self, x, x_diff, y=None, phase='val'):
        x, z, branch = self.forward_features(x, x_diff, y, phase)

        x = self.head(x)
        branch = self.head2(branch)

        zz = []
        for j in z[0]:

            zz.append(torch.softmax(j, dim=-1))

        return x, zz, branch


class Trans_Pos_101(nn.Module):

    def __init__(
            self, img_size=224, patch_size=4, in_chans=2048, num_classes=1004, embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), wss=None,
            block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        
        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        # classification head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x, y=None, phase='val'):
        B = x.shape[0]
        z=[]
        output = None

        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x, z2, branch = blk(x, output)

                z.append(z2)
                if j == 0:
                    x = pos_blk(x)

        del output
        x = self.norm(x)

        return x.mean(dim=1), z, x.mean(dim=1)

    def forward(self, x, y=None, phase='val'):
        x, z, branch = self.forward_features(x, y, phase)

        x = self.head(x)

        zz = []
        for j in z[0]:

            zz.append(torch.softmax(j, dim=-1))

        return x, zz, branch


def _create_twins_pos(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        Trans_Pos, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)
    return model


def _create_twins_pos_101(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        Trans_Pos_101, variant, pretrained,
        default_cfg=default_cfgs[variant],
        **kwargs)
    return model


def twins_zijida_pos(pretrained=False, **kwargs): # embed_dims=[2048,1024]
    model_kwargs = dict(
        patch_size=4, embed_dims=[2048], num_heads=[8], mlp_ratios=[8],
        depths=[1], sr_ratios=[8], **kwargs)
    return _create_twins_pos('aaa', pretrained=pretrained, block_cls = Block_MoE_v8, **model_kwargs)


def twins_zijida_pos_101(pretrained=False, **kwargs): # embed_dims=[2048,1024]
    model_kwargs = dict(
        patch_size=4, embed_dims=[2048], num_heads=[8], mlp_ratios=[8],
        depths=[1], sr_ratios=[8], **kwargs)
    return _create_twins_pos_101('aaa', pretrained=pretrained, block_cls = Block_MoE_v8, **model_kwargs)


if __name__ == '__main__':
    aaa = twins_zijida_pos_101()
    input = torch.randn(7,150,2048)
    output = aaa.forward(input)
