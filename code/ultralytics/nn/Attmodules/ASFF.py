import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion (ASFF)
    论文: Learning Spatial Fusion for Single-Shot Object Detection (AAAI 2020)

    输入 xs = [x0, x1, x2]（P3/P4/P5 三个尺度，分辨率从大到小）
    输出 (B, C, H_l, W_l) 融合特征，分辨率对齐到 level 对应尺度。

    核心思想：
      - 三个输入各过 1x1 卷积压缩通道 -> inter_dim
      - 用 F.interpolate 把三路对齐到 level 对应的分辨率 (scale_factor = 2^(k-level))
      - 每路再过 1x1 卷积生成空间权重图，沿 level 维 softmax -> alpha,beta,gamma
        （每个位置/通道权重和为 1，网络自行决定该位置信哪个尺度）
      - 加权求和后过 3x3 卷积精炼输出
    """

    def __init__(self, level: int, in_channels: list, out_channels: int):
        """
        Args:
            level: 输出尺度索引 0/1/2，对应 P3/P4/P5
            in_channels: 三个输入特征的通道数 [c_P3, c_P4, c_P5]
            out_channels: 融合输出通道数（解析时已按 scale 缩放）
        """
        super().__init__()
        self.level = level
        c0, c1, c2 = in_channels
        self.compress = nn.ModuleList([self._conv(c, out_channels) for c in (c0, c1, c2)])  # 通道压缩
        self.weight = nn.ModuleList([self._conv(out_channels, out_channels) for _ in range(3)])  # 权重图
        self.softmax = nn.Softmax(dim=1)  # 沿 level 维归一化
        self.fuse = self._conv(out_channels, out_channels, k=3)  # 3x3 精炼

    @staticmethod
    def _conv(c1, c2, k=1):
        return nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, xs):
        """
        xs: [x0, x1, x2]，尺度从大到小
        """
        l = self.level

        # ---- Step 1: 通道压缩 ----
        xs = [self.compress[k](x) for k, x in enumerate(xs)]

        # ---- Step 2: 分辨率对齐到 level 对应尺度 ----
        aligned = []
        for k, x in enumerate(xs):
            sf = 2.0 ** (k - l)  # k>l 上采样，k<l 下采样
            if sf != 1.0:
                x = F.interpolate(x, scale_factor=sf, mode="nearest")
            aligned.append(x)

        # ---- Step 3: 生成空间权重图并沿 level 维 softmax 归一化 ----
        ws = torch.stack([self.weight[k](aligned[k]) for k in range(3)], dim=1)  # (B, 3, C, H, W)
        ws = self.softmax(ws)

        # ---- Step 4: 加权求和 + 3x3 精炼 ----
        out = aligned[0] * ws[:, 0] + aligned[1] * ws[:, 1] + aligned[2] * ws[:, 2]
        return self.fuse(out)
