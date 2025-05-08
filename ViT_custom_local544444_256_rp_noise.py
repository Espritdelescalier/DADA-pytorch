import torch
import torch.nn as nn
import math
import numpy as np
from ViT_helper import DropPath, to_2tuple, trunc_normal_
from diff_aug import DiffAugment
import torch.utils.checkpoint as checkpoint
import einops

# "long" and "short" denote longer and shorter samples


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view(
            [batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """

    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view(
            [batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1@x2
        return x


def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                   * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
#         self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))
#         self.noise_strength_2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        #         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
#         x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_2
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16, num_classes=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # print(f"dim {dim}, qk_scale {qk_scale}")
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size

        if self.window_size != 0:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            # torch.stack(torch.meshgrid(coords_h))  # 2, Wh, Ww
            coords = coords_h
            coords_flatten = coords_h  # torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, None] - \
                coords_flatten[None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.contiguous()  # permute(
            # 1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :] += window_size - 1  # shift to start from 0
            # relative_coords[:, :, 1] += window_size - 1
            # relative_coords[:, :, 0] *= 2 * window_size - 1
            # .sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index = relative_coords
            self.register_buffer("relative_position_index",
                                 relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1],
                            device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.window_size != 0:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size, self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            """print(
                f"attn {attn.shape}, relative_position_bias {relative_position_bias.shape}")"""
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, -1)
        # print(torch.max(x))
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(torch.max(x))
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16, num_classes=0):
        super().__init__()
        self.num_classes = num_classes
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, num_classes=num_classes)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print("inside block ", torch.max(x))
        # print(x.shape)
        x = self.drop_path(self.attn(self.norm1(x))) + \
            x
        # print("inside block after attention ", torch.max(x))
        x = self.mlp(self.norm2(x))
        # print("inside block after mlp ", torch.max(x))
        x = x + self.drop_path(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16, num_classes=0):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=window_size,
            num_classes=num_classes
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        #         for blk in self.block:
        #             # x = blk(x)
        #             checkpoint.checkpoint(blk, x)
        #         x = checkpoint.checkpoint(self.block, x)
        x = self.block(x)
        return x


def pixel_upsample(x, H, factor):
    B, N, C = x.size()
    x = x.permute(0, 2, 1)
    x = PixelShuffle1D(factor)(x)  # PixelShuffle1d
    B, C, H = x.size()
    x = x.permute(0, 2, 1)
    return x, H


def bicubic_upsample(x, H, factor):
    B, N, C = x.size()
    x = x.permute(0, 2, 1)
    # x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=factor, mode='linear')
    B, C, H = x.size()
    # x = x.view(-1, C, H*W)
    x = x.permute(0, 2, 1)
    return x, H


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, C = x.shape
    x = x.view(B, H // window_size, window_size, C)
    windows = x.contiguous(
    ).view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, H):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H / window_size))
    x = windows.view(B, H // window_size, window_size, -1)
    x = x.contiguous().view(B, H, -1)
    return x


class Generator(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 logits=True):
        super(Generator, self).__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.latent_dim
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        self.num_classes = args.num_classes
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        self.l2_size = 0
        self.out_act = nn.Tanh()
        self.factor = args.growth_factor
        self.logits = logits

        if self.l2_size == 0:
            self.l1 = nn.Linear(
                args.latent_dim, (self.bottom_width) * self.embed_dim)
            self.lgen_y = nn.Linear(1, self.embed_dim)
        elif self.l2_size > 1000:
            self.l1 = nn.Linear(
                args.latent_dim, (self.bottom_width**2) * self.l2_size//16)
            self.l2 = nn.Sequential(
                nn.Linear(self.l2_size//16, self.l2_size),
                nn.Linear(self.l2_size, self.embed_dim)
            )
        else:
            self.l1 = nn.Linear(
                args.latent_dim, (self.bottom_width) * self.l2_size)
            self.l2 = nn.Linear(self.l2_size, self.embed_dim)
        self.pos_embed_1 = nn.Parameter(
            torch.zeros(1, self.bottom_width, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(
            1, (self.bottom_width)*(args.growth_factor**1), embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(
            1, (self.bottom_width)*(args.growth_factor**2), embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(
            1, (self.bottom_width)*(args.growth_factor**3), embed_dim//(self.factor)))
        self.pos_embed_5 = nn.Parameter(torch.zeros(
            1, (self.bottom_width)*(args.growth_factor**4), embed_dim//(self.factor**2)))
        self.pos_embed_6 = nn.Parameter(torch.zeros(
            1, (self.bottom_width)*(args.growth_factor**5), embed_dim//(self.factor**3)))

        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6
        ]

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]
        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.bottom_width,
            num_classes=self.num_classes
        )
        self.blocks_2 = StageBlock(
            depth=depth[1],
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.bottom_width *
            (args.growth_factor**1),
            num_classes=self.num_classes
        )
        self.blocks_3 = StageBlock(
            depth=depth[2],
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.bottom_width *
            (args.growth_factor**2),
            num_classes=self.num_classes
        )
        self.blocks_4 = StageBlock(
            depth=depth[3],
            dim=embed_dim//(self.factor),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
            num_classes=self.num_classes
        )
        self.blocks_5 = StageBlock(
            depth=depth[4],
            dim=embed_dim//(self.factor**2),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
            num_classes=self.num_classes
        )
        self.blocks_6 = StageBlock(
            depth=depth[5],
            dim=embed_dim//(self.factor**3),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer,
            window_size=self.window_size,
            num_classes=self.num_classes
        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)
        """self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//64, 3, 1, 1, 0)
        )"""
        self.deconv = nn.Sequential(
            nn.Linear(self.embed_dim//(self.factor**3), args.n_channels_g)
        )
        # self.apply(self._init_weights)

    def forward(self, z, gen_y):
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        if self.l2_size == 0:
            x = self.l1(z).view(-1, self.bottom_width, self.embed_dim)
            # gen_y = self.lgen_y(gen_y[:, :, None])
        elif self.l2_size > 1000:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size//16)
            x = self.l2(x)
        else:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
            x = self.l2(x)

        gen_y = gen_y[:, None, :]
        """x = torch.cat(
            [x + self.pos_embed[0], gen_y.float()*torch.ones(x.shape[0], x.shape[1], gen_y.shape[2]).cuda()], dim=-1)"""
        x = x + self.pos_embed[0]
        x[:, :, -self.num_classes:] = gen_y.float()
        # *torch.ones(x.shape[0],x.shape[1], gen_y.shape[2]).cuda()
        B = x.size()
        H = self.bottom_width
        x = self.blocks_1(x)

        x, H = bicubic_upsample(x, H, self.factor)
        x = x + self.pos_embed[1]
        x[:, :, -self.num_classes:] = gen_y.float()
        # *torch.ones(x.shape[0],x.shape[1], gen_y.shape[2]).cuda()
        # x = torch.cat([x + self.pos_embed[1], gen_y], dim=-1)
        B, _, C = x.size()
        x = self.blocks_2(x)

        x, H = bicubic_upsample(x, H, self.factor)
        x = x + self.pos_embed[2]
        x[:, :, -self.num_classes:] = gen_y.float()
        # *torch.ones(x.shape[0],x.shape[1], gen_y.shape[2]).cuda()
        #x = torch.cat([x + self.pos_embed[2], gen_y], dim=-1)
        B, _, C = x.size()
        x = self.blocks_3(x)

        x, H = pixel_upsample(x, H, self.factor)
        x = x + self.pos_embed[3]
        B, _, C = x.size()
        x = x.view(B, H, C)
        x = window_partition(x, self.window_size)
        gen_y = gen_y.float()
        # * \torch.ones(x.shape[0], x.shape[1], gen_y.shape[2]).cuda()
        """print("gen_y before repeat", gen_y.shape)
        gen_y = einops.repeat(gen_y, 'b n c -> (repeat b) n c', repeat=2)
        print("gen_y after repeat", gen_y.shape)
        print("x shape before gen_y cat", x.shape)"""
        x[:, :, -self.num_classes:] = gen_y
        # x = torch.cat([x, gen_y], dim=-1)
        x = self.blocks_4(x)
        x = window_reverse(x, self.window_size, H).view(B, H, C)

        x, H = pixel_upsample(x, H, self.factor)
        x = x + self.pos_embed[4]
        B, _, C = x.size()
        x = x.view(B, H, C)
        x = window_partition(x, self.window_size)
        gen_y = gen_y.float()
        # * \torch.ones(x.shape[0]//2, x.shape[1], gen_y.shape[2]).cuda()
        gen_y = einops.repeat(gen_y, 'b n c -> (repeat b) n c', repeat=2)
        x[:, :, -self.num_classes:] = gen_y
        # x = torch.cat([x, gen_y], dim=-1)
        x = self.blocks_5(x)
        x = window_reverse(x, self.window_size, H).view(B, H, C)

        x, H = pixel_upsample(x, H, self.factor)
        x = x + self.pos_embed[5]
        B, _, C = x.size()
        x = window_partition(x, self.window_size)
        gen_y = gen_y.float()
        # * \torch.ones(x.shape[0]//2, x.shape[1], gen_y.shape[2]).cuda()
        gen_y = einops.repeat(gen_y, 'b n c -> (repeat b) n c', repeat=2)
        x[:, :, -self.num_classes:] = gen_y
        """x[:, :, -self.num_classes:] = gen_y.float()*torch.ones(x.shape[0],
                                                               x.shape[1], gen_y.shape[2]).cuda()"""
        # x = torch.cat([x, gen_y], dim=-1)
        x = self.blocks_6(x)
        x = window_reverse(x, self.window_size, H).view(
            B, H, C)
        # print("x shape at end of gen", x.shape)

        output = self.deconv(x)
        if self.logits:
            # print("act")
            return output.permute(0, 2, 1)
            # output = self.out_act(output)

        return self.out_act(output.permute(0, 2, 1))


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


def _1Ddownsample(x, scale_factor=4):
    return nn.AvgPool1d(kernel_size=scale_factor)(x)


class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm, separate=0, window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x*self.gain + self.drop_path(self.attn(self.norm1(x)))*self.gain
        x = x*self.gain + self.drop_path(self.mlp(self.norm2(x)))*self.gain
        return x


class Discriminator(nn.Module):
    def __init__(self, args, img_size=32, patch_size=None, in_chans=3, num_classes=1, embed_dim=None, depth=7, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = args.num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim

        depth = args.d_depth
        self.args = args
        self.patch_size = patch_size = args.patch_size
        norm_layer = args.d_norm
        act_layer = args.d_act
        self.window_size = args.d_window_size

        # change from 2D to 1D to get adequate dim compat
        """self.fRGB_1 = nn.Conv2d(
            in_chans, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_2 = nn.Conv2d(
            in_chans, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_3 = nn.Conv2d(
            in_chans, embed_dim//4, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_4 = nn.Conv2d(
            in_chans, embed_dim//2, kernel_size=patch_size, stride=patch_size, padding=0)"""

        self.fRGB_1 = nn.Conv1d(
            args.channels, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_2 = nn.Conv1d(
            args.channels, embed_dim//8, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_3 = nn.Conv1d(
            args.channels, embed_dim//4, kernel_size=patch_size, stride=patch_size, padding=0)
        self.fRGB_4 = nn.Conv1d(
            args.channels, embed_dim//2, kernel_size=patch_size, stride=patch_size, padding=0)

        num_patches_1 = (args.img_size // patch_size)
        num_patches_2 = ((args.img_size//2) // patch_size)
        num_patches_3 = ((args.img_size//4) // patch_size)
        num_patches_4 = ((args.img_size//8) // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(
            torch.zeros(1, num_patches_1, embed_dim//8))
        self.pos_embed_2 = nn.Parameter(
            torch.zeros(1, num_patches_2, embed_dim//4))
        self.pos_embed_3 = nn.Parameter(
            torch.zeros(1, num_patches_3, embed_dim//2))
        self.pos_embed_4 = nn.Parameter(
            torch.zeros(1, num_patches_4, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_1 = nn.ModuleList([
            DisBlock(
                dim=embed_dim//8, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer, window_size=self.window_size)
            for i in range(depth+1)])
        self.blocks_2 = nn.ModuleList([
            DisBlock(dim=embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer, window_size=0)
            for i in range(depth)])
        self.blocks_3 = nn.ModuleList([
            DisBlock(dim=embed_dim//2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer, window_size=0)
            for i in range(depth)])
        self.blocks_4 = nn.ModuleList([
            DisBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer, window_size=0)
            for i in range(depth)])
        self.last_block = nn.Sequential(
            #             Block(
            #                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            #                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer),
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[0], act_layer=act_layer, norm_layer=norm_layer, window_size=0)
        )

        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(
            embed_dim, 2*self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.activation = nn.Softmax(dim=-1)

        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.pos_embed_4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        if "None" not in self.args.diff_aug:
            x = DiffAugment(x, self.args.diff_aug, True)

        x_1 = self.fRGB_1(x).permute(0, 2, 1)  # .flatten(2).permute(0, 2, 1)
        # print(f"x1 {x_1.shape}")
        x_2 = self.fRGB_2(nn.AvgPool1d(2)(x)).permute(
            0, 2, 1)  # .flatten(2).permute(0, 2, 1)
        # print(f"x2 {x_2.shape}")
        x_3 = self.fRGB_3(nn.AvgPool1d(4)(x)).permute(
            0, 2, 1)  # .flatten(2).permute(0, 2, 1)
        # print(f"x3 {x_3.shape}")
        x_4 = self.fRGB_4(nn.AvgPool1d(8)(x)).permute(
            0, 2, 1)  # .flatten(2).permute(0, 2, 1)
        # print(f"x4 {x_4.shape}")
        B = x.shape[0]

        x = x_1 + self.pos_embed_1
        x = self.pos_drop(x)
        H = self.args.img_size // self.patch_size
        B, _, C = x.size()
        # x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        # x = x.view(-1, self.window_size*self.window_size, C)
        for blk in self.blocks_1:
            x = blk(x)
        x = x.view(-1, self.window_size, C)
        x = window_reverse(x, self.window_size, H)  # .view(B, H, C)

        _, _, C = x.shape
        x = x.permute(0, 2, 1)  # .view(B, C, H)
#         x = SpaceToDepth(2)(x)
        x = nn.AvgPool1d(2)(x)
        _, _, H = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_2], dim=-1)
        x = x + self.pos_embed_2

        for blk in self.blocks_2:
            x = blk(x)

        _, _, C = x.shape
        x = x.permute(0, 2, 1)  # .view(B, C, H)
#         x = SpaceToDepth(2)(x)
        x = nn.AvgPool1d(2)(x)
        _, _, H = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_3], dim=-1)
        x = x + self.pos_embed_3

        for blk in self.blocks_3:
            x = blk(x)

        _, _, C = x.shape
        x = x.permute(0, 2, 1)  # .view(B, C, H)
#         x = SpaceToDepth(2)(x)
        x = nn.AvgPool1d(2)(x)
        _, _, H = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_4], dim=-1)
        x = x + self.pos_embed_4

        for blk in self.blocks_4:
            x = blk(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, features=False):
        x = self.forward_features(x)
        output = self.head(x)
        output = self.activation(output)
        if features:
            # print("features")
            return output, x
        return output
