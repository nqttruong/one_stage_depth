"""OGCDE model: YOLOv8-style backbone + PAN-FPN neck + extended geometry head.

Head output channels per spatial location (anchor-free):
    4    : bbox (cx, cy, w, h encoded)
    1    : objectness
    nc   : class logits
    1    : depth  (predicted in log-space, decode via exp)
    1    : scale  (predicted in log-space, decode via exp)
    2    : contact offset (dx, dy) in units of stride

Total = 9 + nc channels per scale.
"""

import torch
import torch.nn as nn


def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Conv-BN-SiLU, the YOLOv8 basic unit."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3)
        self.cv2 = Conv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """YOLOv8 C2f — CSP-like with partial residuals."""

    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Backbone(nn.Module):
    """YOLOv8n-scale backbone. Outputs feature maps at strides 8, 16, 32."""

    def __init__(self, w=(16, 32, 64, 128, 256)):
        super().__init__()
        self.stem = Conv(3, w[0], 3, 2)                          # P1/2
        self.dark2 = nn.Sequential(Conv(w[0], w[1], 3, 2), C2f(w[1], w[1], 1, True))   # P2/4
        self.dark3 = nn.Sequential(Conv(w[1], w[2], 3, 2), C2f(w[2], w[2], 2, True))   # P3/8
        self.dark4 = nn.Sequential(Conv(w[2], w[3], 3, 2), C2f(w[3], w[3], 2, True))   # P4/16
        self.dark5 = nn.Sequential(Conv(w[3], w[4], 3, 2), C2f(w[4], w[4], 1, True), SPPF(w[4], w[4]))  # P5/32
        self.out_ch = (w[2], w[3], w[4])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5


class Neck(nn.Module):
    """PAN-FPN: top-down then bottom-up."""

    def __init__(self, ch=(64, 128, 256)):
        super().__init__()
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c4_up = C2f(c4 + c5, c4, 1)
        self.c3_up = C2f(c3 + c4, c3, 1)
        self.d3 = Conv(c3, c3, 3, 2)
        self.c4_dn = C2f(c3 + c4, c4, 1)
        self.d4 = Conv(c4, c4, 3, 2)
        self.c5_dn = C2f(c4 + c5, c5, 1)
        self.out_ch = ch

    def forward(self, p3, p4, p5):
        p4_ = self.c4_up(torch.cat([self.up(p5), p4], 1))
        p3_ = self.c3_up(torch.cat([self.up(p4_), p3], 1))
        p4__ = self.c4_dn(torch.cat([self.d3(p3_), p4_], 1))
        p5__ = self.c5_dn(torch.cat([self.d4(p4__), p5], 1))
        return p3_, p4__, p5__  # strides 8, 16, 32


class OGCDEHead(nn.Module):
    """Extended anchor-free head.

    For each spatial location, predicts 9 + nc channels:
        [cx_raw, cy_raw, w_raw, h_raw,
         obj_logit,
         cls_logit * nc,
         d_raw,      # log-depth
         s_raw,      # log-scale
         dx_raw, dy_raw]   # contact offset in units of stride
    """

    N_GEOM = 4  # d + s + dx + dy

    def __init__(self, nc=3, ch=(64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.no = 4 + 1 + nc + self.N_GEOM
        # Separate detection + geometry sub-heads can help, but a single
        # conv keeps things simple and matches the spec.
        self.heads = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, self.no, 1),
            )
            for c in ch
        )
        # Lightly bias objectness toward negative so training starts stable
        for m in self.heads:
            final = m[-1]
            nn.init.constant_(final.bias[4], -2.0)  # obj logit ~ sigmoid(-2) ≈ 0.12

    def forward(self, feats):
        return [h(f) for h, f in zip(self.heads, feats)]


class OGCDENet(nn.Module):
    """Full OGCDE one-stage model.

    Args:
        nc: number of classes.
        width: tuple of backbone channel widths. Default is YOLOv8n-scale.
    """

    STRIDES = (8, 16, 32)

    def __init__(self, nc=3, width=(16, 32, 64, 128, 256)):
        super().__init__()
        self.nc = nc
        self.backbone = Backbone(width)
        self.neck = Neck(self.backbone.out_ch)
        self.head = OGCDEHead(nc=nc, ch=self.neck.out_ch)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return self.head([f3, f4, f5])

    def load_pretrained_backbone(self, yolov8n_path: str) -> None:
        """Transplant YOLOv8n COCO backbone weights into self.backbone.

        Layer shapes are identical (both YOLOv8n-scale, width=(16,32,64,128,256)).
        ultralytics is only needed here — not at inference time.
        """
        from ultralytics import YOLO
        src = YOLO(yolov8n_path).model.state_dict()
        prefix_map = {
            "stem":    "model.0",
            "dark2.0": "model.1", "dark2.1": "model.2",
            "dark3.0": "model.3", "dark3.1": "model.4",
            "dark4.0": "model.5", "dark4.1": "model.6",
            "dark5.0": "model.7", "dark5.1": "model.8", "dark5.2": "model.9",
        }
        mapped = {}
        for ogcde_pfx, yolo_pfx in prefix_map.items():
            for k, v in src.items():
                if k.startswith(yolo_pfx + "."):
                    mapped[ogcde_pfx + k[len(yolo_pfx):]] = v
        self.backbone.load_state_dict(mapped, strict=True)
        print(f"[pretrained] loaded {len(mapped)} backbone tensors from {yolov8n_path}")


# ----------------------------- decoding helpers ------------------------------

def split_pred(p, nc):
    """Split raw head output (B, no, H, W) -> component tensors.

    Returns a dict where each entry has shape (B, H*W, C_component) after
    flattening spatial dims.
    """
    B, _, H, W = p.shape
    p = p.permute(0, 2, 3, 1).reshape(B, H * W, -1)
    return {
        "bbox_raw": p[..., :4],
        "obj":      p[..., 4],
        "cls":      p[..., 5:5 + nc],
        "d_raw":    p[..., 5 + nc],
        "s_raw":    p[..., 6 + nc],
        "dxdy":     p[..., 7 + nc:9 + nc],
    }


def make_grid(H, W, device):
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([gx, gy], -1).reshape(-1, 2)  # (HW, 2)


def decode_bbox(bbox_raw, grid, stride):
    """YOLOv8-ish decode.

    cx = (sigmoid(bbox_raw[:2]) * 2 - 0.5 + grid) * stride
    wh = (sigmoid(bbox_raw[2:]) * 2) ** 2 * stride

    Using sigmoid*2 instead of exp avoids training blow-up on wh.
    """
    xy = (torch.sigmoid(bbox_raw[..., :2]) * 2 - 0.5 + grid) * stride
    wh = (torch.sigmoid(bbox_raw[..., 2:]) * 2).pow(2) * stride
    return torch.cat([xy, wh], -1)  # (..., 4) cx,cy,w,h


def decode_contact(dxdy_raw, bbox_xywh, stride):
    """Contact point = bbox bottom center + stride * dxdy_raw (direct regression).

    Raw outputs are in units of stride (so at stride=8 an output of 1.0 means
    an 8-pixel offset). Raw outputs are initialized near zero, meaning the
    contact point starts at the bbox bottom center — a reasonable prior.

    No tanh saturation: KITTI contact offsets can be larger than ±2 cells for
    truncated or overhanging objects.
    """
    bx = bbox_xywh[..., 0]
    by = bbox_xywh[..., 1] + bbox_xywh[..., 3] / 2
    off = dxdy_raw * stride
    u = bx + off[..., 0]
    v = by + off[..., 1]
    return torch.stack([u, v], -1)


def decode_depth_scale(d_raw, s_raw):
    """Exp activation for positivity.

    d_pred = exp(d_raw),   s_pred = exp(s_raw),   distance = exp(s_raw + d_raw)
    All three are strictly positive and the log-sum form avoids large
    intermediate magnitudes when computing distance.
    """
    d = torch.exp(d_raw.clamp(min=-5, max=6))   # depth ∈ [~0.007, ~400]
    s = torch.exp(s_raw.clamp(min=-3, max=3))   # scale ∈ [~0.05, ~20]
    dist = torch.exp((s_raw + d_raw).clamp(min=-5, max=6))
    return d, s, dist
