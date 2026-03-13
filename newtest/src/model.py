"""
model.py — Three-Stage Weapon Detection Architecture

Stage 1 : YOLOv8x-OBB  — oriented bounding box detector
           Backbone : CSPDarknet + C2f bottlenecks
           Neck     : PAN-FPN  (multi-scale feature aggregation)
           Head     : Decoupled OBB head  → (cx, cy, w, h, θ, class_logits)

Stage 2 : EfficientNet-B5 — fine-grained weapon classifier
           Input    : detector crops (padded + resized to 224×224)
           Output   : 7-class softmax

Stage 3 : BiLSTM temporal smoother
           Input    : sliding window of per-frame feature vectors
           Output   : smoothed confidence + class per frame
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Utility layers
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    """Conv → BN → SiLU (default). Foundation of all YOLOv8 blocks."""
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard CSP Bottleneck used in C2f."""
    def __init__(self, c, shortcut=True, e=0.5):
        super().__init__()
        hidden = int(c * e)
        self.cv1 = ConvBnAct(c, hidden, 3, 1)
        self.cv2 = ConvBnAct(hidden, c, 3, 1)
        self.use_skip = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.use_skip else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    Cross-Stage Partial bottleneck with 2 convolutions.
    YOLOv8's core building block — replaces C3 from v5.
    """
    def __init__(self, in_c, out_c, n=1, shortcut=False, e=0.5):
        super().__init__()
        hidden = int(out_c * e)
        self.cv1  = ConvBnAct(in_c, 2 * hidden, 1)
        self.cv2  = ConvBnAct((2 + n) * hidden, out_c, 1)
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(hidden, shortcut, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling — Fast version. Used at end of backbone."""
    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        mid_c = in_c // 2
        self.cv1  = ConvBnAct(in_c, mid_c, 1, 1)
        self.cv2  = ConvBnAct(mid_c * 4, out_c, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — YOLOv8x CSPDarknet Backbone
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8Backbone(nn.Module):
    """
    YOLOv8-x CSPDarknet backbone.
    Returns 3 feature maps at strides 8, 16, 32 (P3, P4, P5).
    Channel widths scaled to 'x' variant: width_multiple=1.25, depth_multiple=1.0
    """
    def __init__(self, width_mult=1.25, depth_mult=1.0):
        super().__init__()
        def w(c): return max(round(c * width_mult), 1)
        def d(n): return max(round(n * depth_mult), 1)

        # Stem
        self.p1 = ConvBnAct(3,    w(64),  3, 2)   # stride 2  → /2
        self.p2 = nn.Sequential(
            ConvBnAct(w(64),  w(128), 3, 2),        # stride 2  → /4
            C2f(w(128), w(128), n=d(3), shortcut=True),
        )
        self.p3 = nn.Sequential(
            ConvBnAct(w(128), w(256), 3, 2),         # stride 2  → /8
            C2f(w(256), w(256), n=d(6), shortcut=True),
        )
        self.p4 = nn.Sequential(
            ConvBnAct(w(256), w(512), 3, 2),         # stride 2  → /16
            C2f(w(512), w(512), n=d(6), shortcut=True),
        )
        self.p5 = nn.Sequential(
            ConvBnAct(w(512), w(512), 3, 2),         # stride 2  → /32
            C2f(w(512), w(512), n=d(3), shortcut=True),
            SPPF(w(512), w(512)),
        )

        self.out_channels = [w(256), w(512), w(512)]  # P3, P4, P5

    def forward(self, x):
        x  = self.p1(x)
        x  = self.p2(x)
        p3 = self.p3(x)   # /8
        p4 = self.p4(p3)  # /16
        p5 = self.p5(p4)  # /32
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────────────────
#  PAN-FPN Neck  (Path Aggregation Network)
# ─────────────────────────────────────────────────────────────────────────────

class PANNeck(nn.Module):
    """
    Bidirectional FPN: top-down pathway (FPN) + bottom-up pathway (PAN).
    Fuses P3/P4/P5 backbone features into detection-ready feature maps.
    """
    def __init__(self, in_channels: List[int], width_mult=1.25, depth_mult=1.0):
        super().__init__()
        c3, c4, c5 = in_channels
        def d(n): return max(round(n * depth_mult), 1)

        # ── Top-down (FPN) ────────────────────────────────────────────────
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p4  = C2f(c5 + c4, c4, n=d(3))
        self.td_c2f_p3  = C2f(c4 + c3, c3, n=d(3))

        # ── Bottom-up (PAN) ───────────────────────────────────────────────
        self.bu_conv_p4 = ConvBnAct(c3, c3, 3, 2)
        self.bu_c2f_p4  = C2f(c3 + c4, c4, n=d(3))
        self.bu_conv_p5 = ConvBnAct(c4, c4, 3, 2)
        self.bu_c2f_p5  = C2f(c4 + c5, c5, n=d(3))

        self.out_channels = [c3, c4, c5]

    def forward(self, features):
        p3, p4, p5 = features

        # FPN top-down
        td_p4 = self.td_c2f_p4(torch.cat([self.upsample(p5), p4], dim=1))
        td_p3 = self.td_c2f_p3(torch.cat([self.upsample(td_p4), p3], dim=1))

        # PAN bottom-up
        bu_p4 = self.bu_c2f_p4(torch.cat([self.bu_conv_p4(td_p3), td_p4], dim=1))
        bu_p5 = self.bu_c2f_p5(torch.cat([self.bu_conv_p5(bu_p4),  p5  ], dim=1))

        return td_p3, bu_p4, bu_p5  # strides 8, 16, 32


# ─────────────────────────────────────────────────────────────────────────────
#  OBB Detection Head
# ─────────────────────────────────────────────────────────────────────────────

class OBBHead(nn.Module):
    """
    Decoupled detection head for Oriented Bounding Boxes.
    Per anchor-free grid cell predicts:
      - 4 × DFL regs  (x, y, w, h distribution)
      - 1 angle       (θ ∈ [-π/4, π/4] — predicted via sigmoid then scaled)
      - num_classes   (independent classifier branch)

    DFL (Distribution Focal Loss) regresses each box edge as a discrete
    distribution over 16 bins — more accurate than direct regression.
    """
    DFL_BINS = 16

    def __init__(self, in_channels: List[int], num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = self.DFL_BINS

        # Shared DFL convolution
        self.dfl = nn.Conv2d(4 * self.reg_max, 4, 1, bias=False)
        self.dfl.weight.data[:] = self._make_dfl_weight()

        self.reg_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        self.ang_heads = nn.ModuleList()

        for in_c in in_channels:
            mid_c = max(in_c, 4 * self.reg_max)
            # Regression branch
            self.reg_heads.append(nn.Sequential(
                ConvBnAct(in_c, mid_c, 3),
                ConvBnAct(mid_c, mid_c, 3),
                nn.Conv2d(mid_c, 4 * self.reg_max, 1),
            ))
            # Classification branch
            self.cls_heads.append(nn.Sequential(
                ConvBnAct(in_c, in_c, 3),
                ConvBnAct(in_c, in_c, 3),
                nn.Conv2d(in_c, num_classes, 1),
            ))
            # Angle branch (single scalar per cell)
            self.ang_heads.append(nn.Sequential(
                ConvBnAct(in_c, in_c // 2, 3),
                nn.Conv2d(in_c // 2, 1, 1),
            ))

    def _make_dfl_weight(self):
        """Initialise DFL conv as soft-argmax across bins."""
        w = torch.arange(self.DFL_BINS, dtype=torch.float32)
        w = w.view(1, self.DFL_BINS, 1, 1).expand(4, -1, 1, 1)
        return w.reshape(4 * self.DFL_BINS, 1, 1, 1)

    def forward(self, features: List[torch.Tensor]):
        """
        features: list of (B, C, H, W) at 3 scales.
        Returns list of (B, H*W, 4+1+num_classes) per scale.
        """
        outputs = []
        for i, feat in enumerate(features):
            B, _, H, W = feat.shape

            # ── regression (DFL) ──────────────────────────────────────────
            reg = self.reg_heads[i](feat)  # (B, 4*reg_max, H, W)
            # Compute expected value of each distribution
            reg = reg.view(B, 4, self.reg_max, H, W)
            reg = F.softmax(reg, dim=2)
            reg = self.dfl(reg.view(B, 4 * self.reg_max, H, W))  # (B,4,H,W)

            # ── classification ────────────────────────────────────────────
            cls = self.cls_heads[i](feat)   # (B, num_classes, H, W)

            # ── angle  ────────────────────────────────────────────────────
            # sigmoid → [0,1] → scale to [-π/4, π/4]
            ang = self.ang_heads[i](feat)   # (B, 1, H, W)
            ang = (torch.sigmoid(ang) - 0.5) * (math.pi / 2)

            # ── assemble ──────────────────────────────────────────────────
            # reg: (B,4,H,W) | ang: (B,1,H,W) | cls: (B,nc,H,W)
            out = torch.cat([reg, ang, cls], dim=1)          # (B, 5+nc, H, W)
            out = out.flatten(2).permute(0, 2, 1)            # (B, H*W, 5+nc)
            outputs.append(out)

        return outputs  # 3 × (B, anchors, 5 + num_classes)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Complete YOLOv8x-OBB Detector
# ─────────────────────────────────────────────────────────────────────────────

class WeaponDetector(nn.Module):
    """
    Full YOLOv8x-OBB weapon detector.
    Input  : (B, 3, 640, 640) normalised float32 images
    Output : list of 3 tensors, each (B, H_i*W_i, 5+num_classes)
             [cx, cy, w, h, θ, class_logits...]
    """
    STRIDES = [8, 16, 32]

    def __init__(self, num_classes: int = 7,
                 width_mult: float = 1.25,
                 depth_mult: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = YOLOv8Backbone(width_mult, depth_mult)
        self.neck     = PANNeck(self.backbone.out_channels, width_mult, depth_mult)
        self.head     = OBBHead(self.neck.out_channels, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds = self.head(feats)
        return preds

    def decode_predictions(
        self,
        preds:      List[torch.Tensor],
        img_size:   int = 640,
        conf_thresh: float = 0.35,
        iou_thresh:  float = 0.45,
    ) -> List[torch.Tensor]:
        """
        Decode raw head outputs → final detections per image.
        Returns list (one per batch item) of tensors (N, 7):
            [cx_px, cy_px, w_px, h_px, θ, conf, class_id]
        """
        batch_detections = []
        B = preds[0].shape[0]

        for b in range(B):
            all_boxes  = []
            all_scores = []
            all_angles = []
            all_classes = []

            for stride, pred in zip(self.STRIDES, preds):
                P = pred[b]                          # (H*W, 5+nc)
                H = W = img_size // stride

                # Generate anchor grid
                ys = torch.arange(H, device=P.device).float()
                xs = torch.arange(W, device=P.device).float()
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (H*W, 2)

                # Decode box
                cxcy = (P[:, :2] + grid) * stride    # centre in pixels
                wh   = torch.exp(P[:, 2:4]) * stride  # width/height in pixels
                boxes = torch.cat([cxcy, wh], dim=-1) # (H*W, 4)
                theta = P[:, 4]                        # angle (radians)

                # Class scores
                cls_logits = P[:, 5:]
                scores, cls_ids = F.softmax(cls_logits, dim=-1).max(dim=-1)

                # Objectness = max class prob (no separate obj head in v8)
                keep = scores > conf_thresh
                all_boxes.append(boxes[keep])
                all_scores.append(scores[keep])
                all_angles.append(theta[keep])
                all_classes.append(cls_ids[keep])

            if not all_boxes:
                batch_detections.append(torch.zeros((0, 7), device=preds[0].device))
                continue

            boxes   = torch.cat(all_boxes)
            scores  = torch.cat(all_scores)
            angles  = torch.cat(all_angles)
            classes = torch.cat(all_classes)

            # Rotated NMS (approximate via axis-aligned IoU for speed)
            keep_ids = torchvision_nms(boxes, scores, iou_thresh)
            det = torch.cat([
                boxes[keep_ids],
                angles[keep_ids].unsqueeze(-1),
                scores[keep_ids].unsqueeze(-1),
                classes[keep_ids].float().unsqueeze(-1),
            ], dim=-1)   # (N, 7)
            batch_detections.append(det)

        return batch_detections


def torchvision_nms(boxes, scores, iou_thresh):
    """Axis-aligned NMS via torchvision (fast; acceptable approximation for OBB)."""
    try:
        from torchvision.ops import nms
        # boxes format for torchvision: (x1,y1,x2,y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return nms(torch.stack([x1, y1, x2, y2], dim=-1), scores, iou_thresh)
    except ImportError:
        # Fallback: return all (no NMS)
        return torch.arange(len(scores))


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — EfficientNet-B5 Fine-Grained Classifier
# ─────────────────────────────────────────────────────────────────────────────

class WeaponClassifier(nn.Module):
    """
    EfficientNet-B5 fine-grained weapon classifier.
    Input  : (B, 3, 224, 224) cropped weapon regions
    Output : (B, num_classes) logits

    Architecture changes vs stock EfficientNet-B5:
      - replace classifier head: GlobalAvgPool → Dropout(0.4) → Linear
      - GeM pooling option (better for fine-grained recognition)
    """

    def __init__(self, num_classes: int = 7,
                 dropout: float = 0.4,
                 use_gem: bool = True,
                 pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b5",
                pretrained  = pretrained,
                num_classes = 0,           # remove stock head
                global_pool = "",          # we provide our own pooling
            )
            in_features = self.backbone.num_features   # 2048 for B5
        except ImportError:
            # Fallback: torchvision EfficientNet-B5
            from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
            base = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT
                                   if pretrained else None)
            self.backbone = base.features
            in_features = 2048

        self.pool = GeMPooling(p=3) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # Weight init for new head
        nn.init.kaiming_normal_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        nn.init.xavier_normal_(self.head[5].weight)
        nn.init.zeros_(self.head[5].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled   = self.pool(features)
        return self.head(pooled)

    def freeze_backbone(self):
        """Freeze backbone for first N warm-up epochs."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


class GeMPooling(nn.Module):
    """
    Generalised Mean Pooling — outperforms avg-pool on fine-grained tasks
    by exaggerating discriminative activations.
    p=3 is a common default (p=1 = avg pool, p→∞ = max pool).
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1,
        ).pow(1.0 / self.p)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — BiLSTM Temporal Smoother
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSmoother(nn.Module):
    """
    Bidirectional LSTM that processes a sliding window of per-frame
    detection features and outputs smoothed per-frame predictions.

    Why BiLSTM?  Looking at future frames disambiguates whether a detection
    in the current frame is a genuine start of a weapon appearance or a
    spurious single-frame false positive.

    Input per frame (14-dim feature vector):
      [conf, cx_norm, cy_norm, w_norm, h_norm, θ_norm,
       p0..p6 (classifier softmax),
       motion_magnitude (optical flow magnitude normalised)]

    Output per frame: (smoothed_conf, class_logits[7])
    """

    def __init__(self, input_size: int = 14,
                 hidden_size: int = 128,
                 num_layers:  int = 2,
                 num_classes: int = 7,
                 dropout:     float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional

        # Confidence regressor
        self.conf_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Class predictor
        self.cls_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Layer norm on LSTM output for training stability
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, input_size)  — batch of sliding windows
        Returns:
            conf  : (B, T, 1)   — smoothed confidence per frame
            logits: (B, T, nc)  — class logits per frame
        """
        lstm_out, _ = self.lstm(x)             # (B, T, 2*hidden)
        lstm_out    = self.layer_norm(lstm_out)
        conf   = self.conf_head(lstm_out)       # (B, T, 1)
        logits = self.cls_head(lstm_out)        # (B, T, nc)
        return conf, logits


# ─────────────────────────────────────────────────────────────────────────────
#  Combined inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

class WeaponDetectionPipeline(nn.Module):
    """
    Wraps all three stages into a single module for convenient inference.
    Training each stage is done independently via their own train_*.py scripts.
    """

    CLASSES = ["pistol", "revolver", "rifle", "shotgun",
               "smg", "knife", "blunt_weapon"]

    def __init__(self, cfg: dict):
        super().__init__()
        nc = cfg["dataset"]["num_classes"]
        self.detector   = WeaponDetector(num_classes=nc)
        self.classifier = WeaponClassifier(num_classes=nc)
        self.smoother   = TemporalSmoother(
            input_size  = cfg["temporal"]["input_size"],
            hidden_size = cfg["temporal"]["hidden_size"],
            num_layers  = cfg["temporal"]["num_layers"],
            num_classes = nc,
            dropout     = cfg["temporal"]["dropout"],
        )
        self.cfg = cfg

    def load_weights(self, detector_ckpt: str,
                     classifier_ckpt: str,
                     smoother_ckpt:   str):
        """Load saved weights into each stage."""
        self.detector.load_state_dict(
            torch.load(detector_ckpt, map_location="cpu")["model"], strict=False
        )
        self.classifier.load_state_dict(
            torch.load(classifier_ckpt, map_location="cpu")["model"], strict=False
        )
        self.smoother.load_state_dict(
            torch.load(smoother_ckpt, map_location="cpu")["model"], strict=False
        )

    @torch.no_grad()
    def predict_frame(self, frame_tensor: torch.Tensor,
                      conf_thresh: float = 0.35) -> List[dict]:
        """
        Single-frame inference (no temporal smoothing).
        frame_tensor : (1, 3, 640, 640)
        Returns list of dicts with keys:
            box_obb (cx,cy,w,h,θ), conf, det_class, cls_class,
            cls_conf, corners, geometry
        """
        from src.geometry import extract_geometry

        preds = self.detector(frame_tensor)
        dets  = self.detector.decode_predictions(
            preds,
            img_size    = frame_tensor.shape[-1],
            conf_thresh = conf_thresh,
        )[0]   # first (only) batch item: (N, 7)

        results = []
        for det in dets:
            cx, cy, w, h, theta, conf, cls_id = det.tolist()
            # Crop for classifier
            crop = self._crop_detection(frame_tensor[0], cx, cy, w, h, theta)
            if crop is None:
                continue
            crop_224 = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                     mode="bilinear", align_corners=False)
            cls_logits = self.classifier(crop_224)[0]
            cls_probs  = F.softmax(cls_logits, dim=-1)
            cls_conf, cls_id_refined = cls_probs.max(dim=-1)

            geom = extract_geometry(cx, cy, w, h, theta.item(),
                                    frame_tensor.shape[-1])
            results.append({
                "box_obb":   (cx, cy, w, h, theta),
                "conf":      conf,
                "det_class": self.CLASSES[int(cls_id)],
                "cls_class": self.CLASSES[int(cls_id_refined)],
                "cls_conf":  float(cls_conf),
                "geometry":  geom,
            })
        return results

    def _crop_detection(self, img: torch.Tensor,
                        cx, cy, w, h, theta,
                        pad: float = 0.15) -> Optional[torch.Tensor]:
        """
        Extract a padded crop from the feature map using the OBB centre.
        For simplicity we crop the axis-aligned bounding rectangle of the OBB.
        """
        _, H, W = img.shape
        pad_w = w * pad
        pad_h = h * pad
        x1 = max(0, int(cx - w/2 - pad_w))
        y1 = max(0, int(cy - h/2 - pad_h))
        x2 = min(W, int(cx + w/2 + pad_w))
        y2 = min(H, int(cy + h/2 + pad_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[:, y1:y2, x1:x2]
