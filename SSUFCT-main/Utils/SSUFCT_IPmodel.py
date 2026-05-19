import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from einops import rearrange
from thop import profile
from timm.models.layers import DropPath
import math
from functools import partial

MIN_NUM_PATCHES = 16


class Separable2D(nn.Module):
    def __init__(self, input_filters, output_filters):
        super(Separable2D, self).__init__()
        self.c1 = nn.Conv2d(in_channels=input_filters, out_channels=input_filters, kernel_size=(3, 3), stride=(1, 1), padding=(0,0), groups=input_filters)
        self.bn1 = nn.BatchNorm2d(input_filters)
        self.c2 = nn.Conv2d(in_channels=input_filters, out_channels=output_filters, kernel_size=(1, 1), stride=(1, 1), padding=(0,0))
        self.bn2 = nn.BatchNorm2d(output_filters)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.c2(x)
        x = self.bn2(x)
        out = self.act(x)

        return out


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class Feedforward_bottleneck_CNN(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., wo_dp_conv=False):
        super(Feedforward_bottleneck_CNN, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()])

        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ]
            layers.extend(dp)

        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.heads = num_heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask,  mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FBCN_ViT_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,dropout=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, wo_dp_conv=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Feedforward_bottleneck_CNN = Feedforward_bottleneck_CNN(dim, dim, 1, mlp_ratio, wo_dp_conv)


    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))  # (B, 197, dim)

        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)

        x = self.Feedforward_bottleneck_CNN(x)

        return x


class FBCN_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, num_heads, mlp_ratio, channels=3,
                 dropout=0., emb_dropout=0., drop_path_rate, norm_layer, wo_dp_conv):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2  # 7×7=49

        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(patch_size, channels, dim, norm_layer)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.to_cls_token = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            FBCN_ViT_Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                drop_path=dpr[i], norm_layer=norm_layer, wo_dp_conv=wo_dp_conv)
            for i in range(depth)
        ])

        self.head = nn.Linear(dim, num_classes)
        self.out = nn.Softmax(dim=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        x = self.patch_embed(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.blocks(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        # x = self.to_cls_token(x[:, 0])
        # x = self.head(x)
        # x = self.out(x)
        # x = self.mlp_head(x)

        return x


class CET(nn.Module):
    def __init__(self, num_classes):
        super(CET, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # self.conv2d_features = nn.Sequential(
        #     nn.Conv2d(in_channels=8 * 8, out_channels=64, kernel_size=(3, 3)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )

        self.dsc_conv2d_features = nn.Sequential(
            Separable2D(input_filters=8 * (20-2), output_filters=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.FBCN_ViT = FBCN_ViT(
            image_size=7,
            patch_size=1,
            num_classes=16,
            dim=64,
            depth=1,
            num_heads=4,
            mlp_ratio=1 / 8,
            channels=64,
            dropout=0.1,
            emb_dropout=0.1,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            wo_dp_conv=False
        )

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(64, num_classes)
                                )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c y) h w')
        x = self.dsc_conv2d_features(x)
        x = self.FBCN_ViT(x)
        output = self.fc(x)

        return output

