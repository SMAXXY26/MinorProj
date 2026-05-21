import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
#  Building blocks
# =============================================================================

class ConvBNSiLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = ConvBNSiLU(c, c, 3)
        self.cv2 = ConvBNSiLU(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP-style block used in YOLOv8."""
    def __init__(self, in_c, out_c, n=1, shortcut=True):
        super().__init__()
        mid = out_c // 2
        self.cv1 = ConvBNSiLU(in_c,       out_c, 1)
        self.cv2 = ConvBNSiLU(mid * (n+2), out_c, 1)
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(mid, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        x  = self.cv1(x)
        c  = x.shape[1] // 2
        y  = [x[:, :c], x[:, c:]]
        for m in self.bottlenecks:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c, k=5):
        super().__init__()
        h = c // 2
        self.cv1 = ConvBNSiLU(c, h, 1)
        self.cv2 = ConvBNSiLU(h * 4, c, 1)
        self.mp  = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# =============================================================================
#  Backbone  (YOLOv8-style)
# =============================================================================

class Backbone(nn.Module):
    def __init__(self, w=0.5, d=0.33):
        super().__init__()
        # Channel sizes scaled by width multiplier
        c = [int(x * w) for x in [64, 128, 256, 512, 1024]]

        self.p1 = ConvBNSiLU(3,    c[0], 3, 2)   # /2
        self.p2 = nn.Sequential(
            ConvBNSiLU(c[0], c[1], 3, 2),          # /4
            C2f(c[1], c[1], n=max(1, round(3*d))),
        )
        self.p3 = nn.Sequential(
            ConvBNSiLU(c[1], c[2], 3, 2),          # /8  → P3
            C2f(c[2], c[2], n=max(1, round(6*d))),
        )
        self.p4 = nn.Sequential(
            ConvBNSiLU(c[2], c[3], 3, 2),          # /16 → P4
            C2f(c[3], c[3], n=max(1, round(6*d))),
        )
        self.p5 = nn.Sequential(
            ConvBNSiLU(c[3], c[4], 3, 2),          # /32 → P5
            C2f(c[4], c[4], n=max(1, round(3*d))),
            SPPF(c[4]),
        )
        self.out_channels = [c[2], c[3], c[4]]    # P3, P4, P5

    def forward(self, x):
        x  = self.p1(x)
        x  = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


# =============================================================================
#  Neck  (FPN + PAN)
# =============================================================================

class Neck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels

        # Top-down
        self.up      = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c4   = C2f(c5 + c4, c4, n=1)
        self.td_c3   = C2f(c4 + c3, c3, n=1)

        # Bottom-up
        self.down_c3 = ConvBNSiLU(c3, c3, 3, 2)
        self.bu_c4   = C2f(c3 + c4, c4, n=1)
        self.down_c4 = ConvBNSiLU(c4, c4, 3, 2)
        self.bu_c5   = C2f(c4 + c5, c5, n=1)

        self.out_channels = [c3, c4, c5]

    def forward(self, p3, p4, p5):
        # Top-down path
        td4 = self.td_c4(torch.cat([self.up(p5), p4], 1))
        td3 = self.td_c3(torch.cat([self.up(td4), p3], 1))

        # Bottom-up path
        bu4 = self.bu_c4(torch.cat([self.down_c3(td3), td4], 1))
        bu5 = self.bu_c5(torch.cat([self.down_c4(bu4), p5], 1))

        return td3, bu4, bu5   # small, medium, large


# =============================================================================
#  Detection Head  (decoupled, anchor-free)
# =============================================================================

class DetectionHead(nn.Module):
    def __init__(self, in_c, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max

        # Box regression branch
        self.box_conv = nn.Sequential(
            ConvBNSiLU(in_c, in_c, 3),
            ConvBNSiLU(in_c, in_c, 3),
            nn.Conv2d(in_c, 4 * reg_max, 1),
        )
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBNSiLU(in_c, in_c, 3),
            ConvBNSiLU(in_c, in_c, 3),
            nn.Conv2d(in_c, num_classes, 1),
        )

        # ── Key fix: initialise cls bias to -log((1-p)/p) for p=0.01 ──
        # This means the model starts predicting ~1% probability for every
        # class — not 50% — so it doesn't fire on everything from epoch 1
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_conv[-1].bias, bias_value.item())
        nn.init.normal_(self.cls_conv[-1].weight, std=0.01)
        nn.init.normal_(self.box_conv[-1].weight, std=0.01)

    def forward(self, x):
        box = self.box_conv(x)   # (B, 4*reg_max, H, W)
        cls = self.cls_conv(x)   # (B, num_classes, H, W)
        return box, cls


# =============================================================================
#  WeaponDetector  (full model)
# =============================================================================

class WeaponDetector(nn.Module):
    def __init__(self, num_classes=3, width_mult=0.5, depth_mult=0.33):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = 16

        self.backbone = Backbone(w=width_mult, d=depth_mult)
        self.neck     = Neck(self.backbone.out_channels)

        c3, c4, c5    = self.neck.out_channels
        self.head_s   = DetectionHead(c3, num_classes, self.reg_max)  # small
        self.head_m   = DetectionHead(c4, num_classes, self.reg_max)  # medium
        self.head_l   = DetectionHead(c5, num_classes, self.reg_max)  # large

        # DFL projection vector
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float32))

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        s, m, l    = self.neck(p3, p4, p5)

        out_s = self.head_s(s)
        out_m = self.head_m(m)
        out_l = self.head_l(l)

        return out_s, out_m, out_l   # each = (box, cls) tuple

    def _make_grid(self, h, w, stride, device):
        """Build anchor-free grid for a single scale."""
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # xy offsets at centre of each cell
        xy = (torch.stack([gx, gy], dim=-1) + 0.5) * stride
        return xy.reshape(-1, 2)

    def _dfl(self, x):
        """Distribution Focal Loss decode: softmax + dot product."""
        B, C, H, W = x.shape
        x = x.reshape(B, 4, self.reg_max, H * W)
        x = x.softmax(2)
        x = (x * self.proj.reshape(1, 1, -1, 1)).sum(2)   # (B, 4, H*W)
        return x

    @torch.no_grad()
    def decode_predictions(self, outputs, img_size=640, conf_thresh=0.25):
        """
        Decode multi-scale outputs into boxes.
        outputs: tuple of 3 x (box_feat, cls_feat)
        Returns list[Tensor(N, 6+num_classes)] one per image.
        """
        strides = [img_size // s for s in [80, 40, 20]]  # 8, 16, 32
        B       = outputs[0][0].shape[0]
        results = [[] for _ in range(B)]

        for (box_feat, cls_feat), stride in zip(outputs, strides):
            H, W   = box_feat.shape[2], box_feat.shape[3]
            grid   = self._make_grid(H, W, stride, box_feat.device)

            # Decode boxes via DFL
            ltrb   = self._dfl(box_feat) * stride   # (B, 4, H*W)
            ltrb   = ltrb.permute(0, 2, 1)          # (B, H*W, 4)

            # Convert ltrb (left top right bottom offsets) to xyxy
            x1 = (grid[:, 0] - ltrb[:, :, 0]).clamp(0, img_size)
            y1 = (grid[:, 1] - ltrb[:, :, 1]).clamp(0, img_size)
            x2 = (grid[:, 0] + ltrb[:, :, 2]).clamp(0, img_size)
            y2 = (grid[:, 1] + ltrb[:, :, 3]).clamp(0, img_size)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, H*W, 4)

            # Class scores
            cls_scores = cls_feat.sigmoid()                 # (B, nc, H, W)
            cls_scores = cls_scores.reshape(B, self.num_classes, -1)
            cls_scores = cls_scores.permute(0, 2, 1)        # (B, H*W, nc)
            scores, labels = cls_scores.max(dim=-1)         # (B, H*W)

            for b in range(B):
                mask = scores[b] > conf_thresh
                if mask.sum() == 0:
                    continue
                det = torch.cat([
                    boxes[b][mask],                          # x1,y1,x2,y2
                    scores[b][mask].unsqueeze(1),            # conf
                    labels[b][mask].float().unsqueeze(1),    # class idx
                ], dim=1)
                results[b].append(det)

        # Merge scales and apply NMS per image
        final = []
        for b in range(B):
            if len(results[b]) == 0:
                final.append(torch.zeros((0, 6), device=outputs[0][0].device))
                continue
            dets = torch.cat(results[b], dim=0)
            # Simple NMS
            keep = self._nms(dets[:, :4], dets[:, 4], iou_thresh=0.45)
            final.append(dets[keep])
        return final

    def _nms(self, boxes, scores, iou_thresh=0.45):
        """Basic NMS."""
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort(descending=True)
        keep   = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest   = order[1:]
            ix1    = x1[rest].clamp(min=float(x1[i]))
            iy1    = y1[rest].clamp(min=float(y1[i]))
            ix2    = x2[rest].clamp(max=float(x2[i]))
            iy2    = y2[rest].clamp(max=float(y2[i]))
            inter  = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
            iou    = inter / (areas[i] + areas[rest] - inter).clamp(1e-6)
            order  = rest[iou <= iou_thresh]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


# =============================================================================
#  WeaponClassifier  (unchanged)
# =============================================================================

class WeaponClassifier(nn.Module):
    # Supported backbones and their feature widths
    _BACKBONES = {
        "efficientnet_b5":    (2048, "EfficientNet_B5_Weights",    "efficientnet_b5"),
        "efficientnet_b0":    (1280, "EfficientNet_B0_Weights",    "efficientnet_b0"),
        "mobilenet_v3_small": (576,  "MobileNet_V3_Small_Weights", "mobilenet_v3_small"),
    }

    def __init__(self, num_classes=3, dropout=0.4, pretrained=True,
                 backbone="efficientnet_b5"):
        super().__init__()
        if backbone not in self._BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {list(self._BACKBONES)}"
            )
        self.backbone_name = backbone
        in_features, weights_cls, factory_fn = self._BACKBONES[backbone]

        weights = getattr(models, weights_cls).DEFAULT if pretrained else None
        base    = getattr(models, factory_fn)(weights=weights)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)


# =============================================================================
#  TemporalSmoother — REMOVED
#  The authoritative TemporalSmoother lives in part3.py (has LayerNorm,
#  correct input_size=8). Import from there instead.
# =============================================================================