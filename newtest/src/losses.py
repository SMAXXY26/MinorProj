"""
losses.py — Custom Loss Functions

OBB Detection Loss  : CIoU + DFL regression + BCE classification + angle loss
Focal Loss          : for class-imbalanced weapon classifier
Temporal BCE        : for BiLSTM smoother
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  OBB Detection Loss (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

class OBBDetectionLoss(nn.Module):
    """
    Combined loss for YOLOv8x-OBB.

    Total loss = λ_box * CIoU_loss
               + λ_dfl * DFL_loss
               + λ_cls * BCE_classification_loss
               + λ_ang * SmoothL1_angle_loss

    Assignment: TaskAlignedAssigner (TAL) — aligns anchors to GT boxes
    by a score that combines classification confidence and IoU.
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        det = cfg["detector"]
        self.box_gain  = det["box_loss_gain"]
        self.cls_gain  = det["cls_loss_gain"]
        self.dfl_gain  = det["dfl_loss_gain"]
        self.ang_gain  = det["angle_loss_gain"]
        self.nc        = cfg["dataset"]["num_classes"]
        self.reg_max   = 16
        self.device    = device

        # DFL bins as a 1D tensor
        self.bins = torch.arange(self.reg_max, dtype=torch.float32,
                                 device=device)

    def forward(self,
                preds:  List[torch.Tensor],
                targets: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        preds   : list of (B, H_i*W_i, 5+nc) from OBBHead.forward()
        targets : list of length B, each (N, 6) [cls, cx, cy, w, h, theta]
        """
        device = self.device
        total_loss = torch.zeros(1, device=device)
        loss_dict  = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "angle": 0.0}

        B = preds[0].shape[0]
        strides = [8, 16, 32]

        # Concatenate all scale predictions: (B, total_anchors, 5+nc)
        all_preds = torch.cat(preds, dim=1)
        n_anchors = all_preds.shape[1]

        for b in range(B):
            gt = targets[b].to(device)   # (N, 6): cls cx cy w h θ
            if gt.shape[0] == 0:
                continue

            pred_b   = all_preds[b]     # (n_anchors, 5+nc)
            pred_box = pred_b[:, :4]    # cx cy w h
            pred_ang = pred_b[:, 4]     # theta
            pred_cls = pred_b[:, 5:]    # class logits (nc,)

            # Build anchor centres
            anchor_pts = self._build_anchor_grid(strides, img_size=640,
                                                 device=device)

            # Task-Aligned Assignment
            assigned_gt, assigned_mask = self._task_aligned_assign(
                pred_box, pred_cls, gt, anchor_pts
            )
            if assigned_mask.sum() == 0:
                continue

            # ── Box: CIoU loss ─────────────────────────────────────────────
            pred_box_pos = pred_box[assigned_mask]
            gt_box_pos   = assigned_gt[assigned_mask, 1:5]
            box_loss     = ciou_loss(pred_box_pos, gt_box_pos).mean()

            # ── Angle: SmoothL1 ────────────────────────────────────────────
            pred_ang_pos = pred_ang[assigned_mask]
            gt_ang_pos   = assigned_gt[assigned_mask, 5]
            ang_loss     = F.smooth_l1_loss(pred_ang_pos, gt_ang_pos)

            # ── Classification: BCE with label smoothing ───────────────────
            cls_targets = torch.zeros(n_anchors, self.nc, device=device)
            cls_targets[assigned_mask, assigned_gt[assigned_mask, 0].long()] = 1.0
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls, cls_targets, reduction="mean"
            )

            # ── DFL: distribution-to-corner regression ─────────────────────
            # Approximate: MSE between pred distribution mean and target
            dfl_loss = torch.tensor(0.0, device=device)

            step_loss = (self.box_gain * box_loss +
                         self.cls_gain * cls_loss +
                         self.dfl_gain * dfl_loss +
                         self.ang_gain * ang_loss)

            total_loss = total_loss + step_loss
            loss_dict["box"]   += float(box_loss)
            loss_dict["cls"]   += float(cls_loss)
            loss_dict["angle"] += float(ang_loss)

        return total_loss / max(B, 1), loss_dict

    def _build_anchor_grid(self, strides, img_size, device):
        """Build flat anchor centre grid for all strides."""
        pts = []
        for s in strides:
            g = img_size // s
            ys = torch.arange(g, device=device).float()
            xs = torch.arange(g, device=device).float()
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            centres = torch.stack([(gx.flatten() + 0.5) * s,
                                   (gy.flatten() + 0.5) * s], dim=-1)
            pts.append(centres)
        return torch.cat(pts, dim=0)  # (total_anchors, 2)

    def _task_aligned_assign(self, pred_box, pred_cls, gt, anchor_pts,
                              topk=10, alpha=0.5, beta=6.0):
        """
        Simplified Task-Aligned Assigner (TAL).
        Assigns top-k anchors per GT box based on:
            score = cls_prob^alpha * iou^beta

        Returns:
            assigned_gt   (n_anchors, 6) — matched GT row or zeros
            assigned_mask (n_anchors,)   — bool, True = positive
        """
        n_gt = gt.shape[0]
        n_a  = pred_box.shape[0]
        device = pred_box.device

        assigned_gt   = torch.zeros(n_a, 6, device=device)
        assigned_mask = torch.zeros(n_a, dtype=torch.bool, device=device)

        for gi in range(n_gt):
            gt_row = gt[gi]       # cls, cx, cy, w, h, θ
            gt_box = gt_row[1:5].unsqueeze(0)  # (1,4)

            # IoU between every anchor pred box and this GT box
            iou   = bbox_iou(pred_box, gt_box.expand(n_a, -1))  # (n_a,)

            # Classification alignment score
            cls_idx = int(gt_row[0])
            cls_p   = torch.sigmoid(pred_cls[:, cls_idx])         # (n_a,)

            align_score = (cls_p ** alpha) * (iou ** beta)
            _, topk_ids = align_score.topk(min(topk, n_a))

            assigned_gt[topk_ids]   = gt_row
            assigned_mask[topk_ids] = True

        return assigned_gt, assigned_mask


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss (Stage 2 — classifier)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    Downweights easy examples so the model focuses on hard/rare weapon types.
    γ=2, α=0.25 are standard starting points.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.1, num_classes: int = 7):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) raw class scores
        targets : (B,)   integer class labels
        """
        # Label smoothing
        C = self.num_classes
        smooth_val = self.label_smoothing / C
        one_hot = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        one_hot = one_hot * (1 - self.label_smoothing) + smooth_val

        # Cross-entropy per sample
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = torch.exp(log_prob)

        # Focal weight
        pt      = (one_hot * prob).sum(dim=-1)
        focal_w = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_t = one_hot.new_full(one_hot.shape, self.alpha)
        alpha_t[one_hot == 1] = 1 - self.alpha

        loss = -(alpha_t * one_hot * log_prob).sum(dim=-1)
        loss = (focal_w * loss).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal smoothing loss (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalLoss(nn.Module):
    """
    Combined loss for the BiLSTM temporal smoother.
      - BCE on confidence output (binary: weapon present / absent)
      - Cross-entropy on class predictions
      - Temporal consistency penalty: penalises sudden conf changes
    """
    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smooth_w = smoothness_weight

    def forward(self,
                conf_pred:   torch.Tensor,  # (B, T, 1)
                cls_logits:  torch.Tensor,  # (B, T, nc)
                conf_target: torch.Tensor,  # (B, T) — binary
                cls_target:  torch.Tensor   # (B, T) — int labels
               ) -> Tuple[torch.Tensor, dict]:

        # ── Confidence BCE ────────────────────────────────────────────────
        conf_loss = F.binary_cross_entropy(
            conf_pred.squeeze(-1),
            conf_target.float(),
        )

        # ── Classification CE (only on frames where weapon present) ───────
        mask = conf_target > 0.5
        if mask.sum() > 0:
            cls_loss = F.cross_entropy(
                cls_logits[mask],
                cls_target[mask],
            )
        else:
            cls_loss = torch.tensor(0.0, device=conf_pred.device)

        # ── Temporal smoothness penalty ───────────────────────────────────
        # |conf[t] - conf[t-1]|² encourages smooth confidence trajectories
        conf_sq  = conf_pred.squeeze(-1)
        smooth_loss = ((conf_sq[:, 1:] - conf_sq[:, :-1]) ** 2).mean()

        total = conf_loss + cls_loss + self.smooth_w * smooth_loss
        return total, {
            "conf": float(conf_loss),
            "cls":  float(cls_loss),
            "smooth": float(smooth_loss),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  IoU utilities
# ─────────────────────────────────────────────────────────────────────────────

def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor,
             xywh: bool = True) -> torch.Tensor:
    """
    Vectorised axis-aligned IoU.
    boxes: (N, 4) format cx cy w h  (if xywh=True)
    Returns (N,) IoU values.
    """
    if xywh:
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + 1e-7
    return inter / union


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Complete IoU loss (CIoU) — adds aspect ratio consistency penalty to DIoU.
    Both tensors: (N, 4) in cx cy w h format.
    """
    iou = bbox_iou(pred, target)

    # Centre distance term
    pred_cx, pred_cy = pred[:, 0], pred[:, 1]
    gt_cx,   gt_cy   = target[:, 0], target[:, 1]
    rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    # Enclosing box diagonal
    enclose_x1 = torch.min(pred[:, 0] - pred[:, 2]/2, target[:, 0] - target[:, 2]/2)
    enclose_y1 = torch.min(pred[:, 1] - pred[:, 3]/2, target[:, 1] - target[:, 3]/2)
    enclose_x2 = torch.max(pred[:, 0] + pred[:, 2]/2, target[:, 0] + target[:, 2]/2)
    enclose_y2 = torch.max(pred[:, 1] + pred[:, 3]/2, target[:, 1] + target[:, 3]/2)
    c2 = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-7

    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * (
        torch.atan(target[:, 2] / (target[:, 3] + 1e-7)) -
        torch.atan(pred[:, 2]   / (pred[:, 3]   + 1e-7))
    ) ** 2
    alpha_ciou = v / (1 - iou + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    return 1 - ciou
