import torch
import torch.nn as nn


class CA(nn.Module):
    """
    Coordinate Attention (CA).

    Input/output: (B, C, H, W) -> (B, C, H, W).

    Outline:
      - global pooling is applied separately along H and along W, which yields two
        direction-aware channel descriptors
      - a shared 1x1 transform splits them into two branches, producing
        Ah (B, C, H, 1) and Aw (B, C, 1, W)
      - the input is re-weighted element-wise by Ah and Aw
    """

    def __init__(self, channels: int, reduction: int = 32):
        """
        Args:
            channels: number of input channels C
            reduction: reduction ratio, 32 in the paper (16 is also used)
        """
        super().__init__()
        assert channels > 0
        mip = max(8, channels // reduction)  # the paper fixes a minimum of 8

        # shared transform: C -> mip (1x1 conv)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        # two branches: mip -> C
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- Step 1: pooling along each axis, which keeps the coordinate information ----
        # x_h: (B, C, H, 1)  averaged along W
        x_h = x.mean(dim=3, keepdim=True)

        # x_w: (B, C, 1, W)  averaged along H
        x_w = x.mean(dim=2, keepdim=True)

        # ---- Step 2: concatenation along the spatial dimension ----
        # x_w is transposed to (B, C, W, 1) so that its last dimension matches x_h
        # x_w_t: (B, C, W, 1)
        x_w_t = x_w.permute(0, 1, 3, 2)

        # y: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w_t], dim=2)

        # ---- Step 3: shared transform (bottleneck) ----
        # y: (B, mip, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # ---- Step 4: split the two branches ----
        # y_h: (B, mip, H, 1)
        # y_w: (B, mip, W, 1)
        y_h, y_w = torch.split(y, [h, w], dim=2)

        # restore y_w to (B, mip, 1, W)
        y_w = y_w.permute(0, 1, 3, 2)  # (B, mip, 1, W)

        # ---- Step 5: build the two directional attention maps ----
        # a_h: (B, C, H, 1)
        a_h = self.sigmoid(self.conv_h(y_h))

        # a_w: (B, C, 1, W)
        a_w = self.sigmoid(self.conv_w(y_w))

        # ---- Step 6: re-weighting (broadcast element-wise product) ----
        # a_h broadcasts along W, a_w broadcasts along H
        out = x * a_h * a_w  # (B, C, H, W)

        return out
