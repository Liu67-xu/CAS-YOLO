import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion (ASFF).

    Paper: Learning Spatial Fusion for Single-Shot Object Detection (AAAI 2020).

    Input  xs = [x0, x1, x2]  (the P3/P4/P5 levels, from the finest to the coarsest)
    Output (B, C, H_l, W_l) fused feature, at the resolution of the target level.

    Outline:
      - each input is compressed to out_channels by a 1x1 convolution
      - F.interpolate aligns all three to the resolution of the target level
        (scale_factor = 2^(k-level))
      - a second 1x1 convolution produces a spatial weight map per level, and a
        softmax over the level dimension gives alpha, beta and gamma (the weights
        at each position sum to one, so the network itself decides which scale to
        trust at that position)
      - the weighted sum is refined by a 3x3 convolution
    """

    def __init__(self, level: int, in_channels: list, out_channels: int):
        """
        Args:
            level: index of the output level, 0/1/2 for P3/P4/P5
            in_channels: channel counts of the three inputs [c_P3, c_P4, c_P5]
            out_channels: channels of the fused output (already scaled by width)
        """
        super().__init__()
        self.level = level
        c0, c1, c2 = in_channels
        self.compress = nn.ModuleList([self._conv(c, out_channels) for c in (c0, c1, c2)])  # channel compression
        self.weight = nn.ModuleList([self._conv(out_channels, out_channels) for _ in range(3)])  # weight maps
        self.softmax = nn.Softmax(dim=1)  # normalised over the level dimension
        self.fuse = self._conv(out_channels, out_channels, k=3)  # 3x3 refinement

    @staticmethod
    def _conv(c1, c2, k=1):
        return nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, xs):
        """
        xs: [x0, x1, x2], from the finest to the coarsest scale.
        """
        l = self.level

        # ---- Step 1: channel compression ----
        xs = [self.compress[k](x) for k, x in enumerate(xs)]

        # ---- Step 2: resample every level to the resolution of the target level ----
        aligned = []
        for k, x in enumerate(xs):
            sf = 2.0 ** (k - l)  # upsample when k > l, downsample when k < l
            if sf != 1.0:
                x = F.interpolate(x, scale_factor=sf, mode="nearest")
            aligned.append(x)

        # ---- Step 3: spatial weight maps, softmax over the level dimension ----
        ws = torch.stack([self.weight[k](aligned[k]) for k in range(3)], dim=1)  # (B, 3, C, H, W)
        ws = self.softmax(ws)

        # ---- Step 4: weighted sum and 3x3 refinement ----
        out = aligned[0] * ws[:, 0] + aligned[1] * ws[:, 1] + aligned[2] * ws[:, 2]
        return self.fuse(out)
