import torch
import torch.nn as nn
import torch.nn.functional as F


class OBBDetectionLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device      = device
        self.num_classes = cfg["dataset"]["num_classes"]
        self.img_size    = cfg["detector"]["img_size"]
        self.reg_max     = 16

    def forward(self, outputs, targets):
        box_loss  = torch.tensor(0.0, device=self.device, requires_grad=True)
        cls_loss  = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_pos     = 0

        strides = [self.img_size // s for s in [
            outputs[0][1].shape[2],   # scale 0 H
            outputs[1][1].shape[2],   # scale 1 H
            outputs[2][1].shape[2],   # scale 2 H
        ]]
        # Recalculate strides from actual feature map sizes
        strides = [self.img_size // box.shape[2] for box, cls in outputs]

        for (box_feat, cls_feat), stride in zip(outputs, strides):
            B, _, H, W = cls_feat.shape

            # ── Build grid ───────────────────────────────────────────────
            gy, gx = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            # Cell centres in pixel coords
            cx_grid = (gx + 0.5) * stride   # (H, W)
            cy_grid = (gy + 0.5) * stride   # (H, W)
            cx_flat = cx_grid.reshape(-1)    # (H*W,)
            cy_flat = cy_grid.reshape(-1)    # (H*W,)

            cls_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, self.num_classes)
            box_flat = box_feat.permute(0, 2, 3, 1).reshape(B, H*W, 4 * self.reg_max)

            # ── Positive/negative mask for classification ────────────────
            # Start with all-negative target (background)
            cls_target_full = torch.zeros(
                B, H*W, self.num_classes, device=self.device
            )
            pos_mask = torch.zeros(B, H*W, dtype=torch.bool, device=self.device)

            for b in range(B):
                gt = targets[b]
                if not isinstance(gt, torch.Tensor):
                    gt = torch.tensor(gt, dtype=torch.float32)
                gt = gt.to(self.device)
                if gt.shape[0] == 0:
                    continue

                gt_cx  = gt[:, 1] * self.img_size   # (T,)
                gt_cy  = gt[:, 2] * self.img_size
                gt_w   = gt[:, 3] * self.img_size
                gt_h   = gt[:, 4] * self.img_size
                gt_cls = gt[:, 0].long().clamp(0, self.num_classes - 1)

                for t in range(gt.shape[0]):
                    # Assign to ALL cells whose centre falls inside GT box
                    x1_t = gt_cx[t] - gt_w[t] / 2
                    y1_t = gt_cy[t] - gt_h[t] / 2
                    x2_t = gt_cx[t] + gt_w[t] / 2
                    y2_t = gt_cy[t] + gt_h[t] / 2

                    inside = (
                        (cx_flat >= x1_t) & (cx_flat <= x2_t) &
                        (cy_flat >= y1_t) & (cy_flat <= y2_t)
                    )

                    if inside.sum() == 0:
                        # Fallback: assign to nearest cell
                        dist = (cx_flat - gt_cx[t])**2 + (cy_flat - gt_cy[t])**2
                        inside[dist.argmin()] = True

                    pos_mask[b] |= inside
                    cls_target_full[b, inside, gt_cls[t]] = 1.0

                    # ── Box loss for positive cells ───────────────────────
                    pos_idx = inside.nonzero(as_tuple=True)[0]

                    # Decode all positive cells at once: (P, 4*reg_max) → (P, 4)
                    proj = torch.arange(
                        self.reg_max, device=self.device, dtype=torch.float32
                    )
                    box_pos = box_flat[b, pos_idx].reshape(len(pos_idx), 4, self.reg_max)
                    ltrb_pos = (box_pos.softmax(-1) * proj).sum(-1) * stride  # (P, 4)

                    # Convert ltrb → xyxy for IoU computation
                    cx_pos = cx_flat[pos_idx]
                    cy_pos = cy_flat[pos_idx]
                    pred_x1 = (cx_pos - ltrb_pos[:, 0]).clamp(0)
                    pred_y1 = (cy_pos - ltrb_pos[:, 1]).clamp(0)
                    pred_x2 = (cx_pos + ltrb_pos[:, 2]).clamp(0)
                    pred_y2 = (cy_pos + ltrb_pos[:, 3]).clamp(0)

                    inter_x1 = torch.maximum(pred_x1, x1_t.expand_as(pred_x1))
                    inter_y1 = torch.maximum(pred_y1, y1_t.expand_as(pred_y1))
                    inter_x2 = torch.minimum(pred_x2, x2_t.expand_as(pred_x2))
                    inter_y2 = torch.minimum(pred_y2, y2_t.expand_as(pred_y2))
                    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
                    area_pred = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
                    area_gt   = gt_w[t] * gt_h[t]
                    with torch.no_grad():
                        ious = inter / (area_pred + area_gt - inter + 1e-7)
                        k = min(4, len(pos_idx))
                        top_k = ious.topk(k).indices
                    pos_idx_filtered = pos_idx[top_k]

                    # ltrb targets for the filtered cells
                    ltrb_target = torch.stack([
                        (cx_flat[pos_idx_filtered] - x1_t).clamp(0),
                        (cy_flat[pos_idx_filtered] - y1_t).clamp(0),
                        (x2_t - cx_flat[pos_idx_filtered]).clamp(0),
                        (y2_t - cy_flat[pos_idx_filtered]).clamp(0),
                    ], dim=1)  # (k, 4)

                    box_loss = box_loss + F.smooth_l1_loss(
                        ltrb_pos[top_k], ltrb_target
                    )
                    n_pos += k

            # ── Classification loss ───────────────────────────────────────
            # Positive cells: BCE with GT class
            # Negative cells: BCE pushing all classes to 0
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                cls_flat.reshape(-1, self.num_classes),
                cls_target_full.reshape(-1, self.num_classes),
                pos_weight=torch.full(
                    (self.num_classes,), 5.0, device=self.device
                ),  # up-weight positives since negatives dominate
            )

        n = max(n_pos, 1)
        # Weight cls higher to fix saturation
        total = box_loss / n + cls_loss * 2.0

        return total, {
            "box":   float(box_loss) / n,
            "cls":   float(cls_loss),
            "angle": 0.0,
        }


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25,
                 label_smoothing=0.1, num_classes=3,
                 class_weights=None):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes
        self.class_weights   = class_weights  # tensor of shape (num_classes,)

    def forward(self, logits, targets):
        ce   = F.cross_entropy(
            logits, targets,
            weight          = self.class_weights,
            label_smoothing = self.label_smoothing,
            reduction       = "none",
        )
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# =============================================================================
#  TemporalLoss — REMOVED
#  The authoritative TemporalLoss lives in part3.py (different interface:
#  takes (conf_pred, cls_logits, labels) and derives targets internally).
# =============================================================================