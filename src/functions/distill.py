"""
distill.py — Production Knowledge Distillation: teachers → YOLOv8n student.
===========================================================================
Implements a three-term KD loss with **multi-teacher ensemble** support:

  1. Response-based (head) — ensemble average over teachers:
       t_cls = mean_k( teacher_k.cls_logits )
       t_dfl = mean_k( teacher_k.dfl_logits )
       cls_kd  = KL( softmax(s_cls/T)  ‖  softmax(t_cls/T) ) · T²
       box_kd  = KL( softmax(s_dfl/T)  ‖  softmax(t_dfl/T) ) · T²   × 0.5
       head_kd = cls_kd + 0.5·box_kd

  2. Feature-based (FPN neck — Attention Transfer, averaged over teachers):
       feat_kd = Σ_l || AT(s_feat_l) − mean_k( AT(t_k_feat_l) ) ||₂²
       where AT(F) = F².mean(dim=1, keepdim=True) normalised to unit L2 norm.
       Teacher features are channel-adapted to student width with a 1×1 conv
       per (teacher, level) pair.

  3. Adaptive alpha scheduling:
       α ramps linearly from kd_alpha_warmup (0.0) → kd_alpha over
       kd_warmup_epochs, then stays flat. This lets the student anchor on GT
       detection loss early (avoids teacher-shaped reward hacking), then
       progressively relies on the teacher's soft targets.

Total loss = (1 − α_cur) · det_loss  +  α_cur · head_kd  +  β · feat_kd

Config (config/hyperparams.yaml → student section):
    kd_temperature    : 6.0   # higher = softer teacher logits (3-class: 6 is optimal)
    kd_alpha          : 0.65  # final KD head weight
    kd_alpha_warmup   : 0.0   # starting alpha (ramps up over kd_warmup_epochs)
    kd_warmup_epochs  : 10    # epochs to reach kd_alpha
    kd_feat_beta      : 0.30  # neck-feature AT loss weight
    kd_feat_layers    : 3     # number of neck output layers to distil (P3/P4/P5)

Functions:
    make_kd_trainer(teacher_paths, **kd_kwargs) → DetectionTrainer subclass
    train_student(cfg, teacher_ckpts) → str (path to best.pt)
    export_student(cfg, student_ckpt) → dict
"""

import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from src.functions.monitor import TrainingMonitor
from src.functions.small_object_aug import build_injector_from_cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Attention Transfer helper
# ══════════════════════════════════════════════════════════════════════════════

def _attention_map(feat: torch.Tensor) -> torch.Tensor:
    """
    Spatial attention map (Zagoruyko & Komodakis, 2017).
    feat: [B, C, H, W]  →  [B, 1, H, W]  (L2-normalised across spatial dims)
    """
    am = feat.pow(2).mean(dim=1, keepdim=True)           # [B, 1, H, W]
    am = am.flatten(2)                                    # [B, 1, H*W]
    am = F.normalize(am, p=2, dim=-1)
    return am                                             # [B, 1, H*W]


def _nwd_similarity(pred_xywh: torch.Tensor, gt_xywh: torch.Tensor,
                    constant: float = 12.8) -> torch.Tensor:
    """
    Normalized Wasserstein Distance similarity (Wang et al., 2021).
    Each box is modelled as a 2-D Gaussian with μ=(cx,cy) and σ=(w/2,h/2).
    W₂²(p,g) = ‖μ_p−μ_g‖² + ‖σ_p−σ_g‖²
    NWD(p,g)  = exp(−√W₂² / C)

    Args:
        pred_xywh: [M, 4]  predicted boxes (cx, cy, w, h) in pixel coords
        gt_xywh:   [K, 4]  GT boxes (cx, cy, w, h) in pixel coords
        constant:  C ≈ 12.8 recommended by the authors for pixel-space boxes

    Returns:
        [K, M] NWD similarity matrix  (1 = perfect match, 0 = far apart)
    """
    p_mu  = pred_xywh[:, :2]        # [M, 2]
    p_sig = pred_xywh[:, 2:] / 2   # [M, 2]
    g_mu  = gt_xywh[:, :2]          # [K, 2]
    g_sig = gt_xywh[:, 2:] / 2     # [K, 2]

    d_mu  = (g_mu.unsqueeze(1)  - p_mu.unsqueeze(0)).pow(2).sum(-1)   # [K, M]
    d_sig = (g_sig.unsqueeze(1) - p_sig.unsqueeze(0)).pow(2).sum(-1)  # [K, M]
    w2    = (d_mu + d_sig).clamp(min=0).sqrt()

    return torch.exp(-w2 / constant)


# ══════════════════════════════════════════════════════════════════════════════
#  Channel adapter registry (1×1 conv: teacher_C → student_C)
# ══════════════════════════════════════════════════════════════════════════════

class _ChannelAdapter(nn.Module):
    """
    Thin 1×1 convolution that projects teacher feature channels to the student's
    channel count for the AT loss.  One adapter per FPN level, kept on the same
    device as the models.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ══════════════════════════════════════════════════════════════════════════════
#  KD Criterion — head KD + neck feature AT KD
# ══════════════════════════════════════════════════════════════════════════════

class _KDCriterion:
    """
    Drop-in wrapper around the student's v8DetectionLoss — multi-teacher.

    Three extra KD terms are added:
      • head_kd  — response-based KL on ensemble-averaged class logits + DFL
      • feat_kd  — attention-transfer L2 on ensemble-averaged FPN feature maps
      • α_cur    — linearly scheduled KD weight (warmup → final)

    Teacher features are captured via forward hooks registered on each teacher's
    neck output layers (separate storage per teacher).  Student features are
    captured similarly.  A per-(teacher, level) 1×1 conv adapts channel widths
    when teacher ≠ student.

    Exposes the same interface as the underlying criterion: __call__(preds, batch).
    """

    def __init__(
        self,
        base_criterion,
        teachers:        list,
        T:               float,
        alpha:           float,
        alpha_warmup:    float,
        warmup_epochs:   int,
        total_epochs:    int,
        feat_beta:       float,
        feat_layers:     int,
        device,
        nwd_weight:      float = 0.0,
        nwd_small_thresh: float = 0.04,
    ):
        self._base           = base_criterion
        self._teachers       = list(teachers)
        self._T              = float(T)
        self._alpha_final    = float(alpha)
        self._alpha_warmup   = float(alpha_warmup)
        self._warmup_epochs  = int(warmup_epochs)
        self._total_epochs   = int(total_epochs)
        self._feat_beta      = float(feat_beta)
        self._feat_layers    = int(feat_layers)
        self._device         = device
        self._current_epoch  = 0
        self._nwd_weight     = float(nwd_weight)
        self._nwd_small_thresh = float(nwd_small_thresh)

        # Per-teacher feature hook storage: list[list[Tensor]] (one list per teacher)
        self._t_feats_per_teacher: list[list[torch.Tensor]] = [
            [] for _ in self._teachers
        ]
        self._s_feats: list[torch.Tensor] = []
        self._hooks:   list               = []
        # Per-teacher channel adapters: list[list[_ChannelAdapter|None]]
        self._adapters: list[list] = []
        self._adapters_ready = False

        # Register hooks on the teacher and student neck output layers
        self._register_feat_hooks()

    # ── trainer-facing attribute forwarding ───────────────────────────────────
    def update(self):
        if hasattr(self._base, "update"):
            self._base.update()

    @property
    def updates(self):
        return getattr(self._base, "updates", 0)

    @updates.setter
    def updates(self, value):
        if hasattr(self._base, "updates"):
            self._base.updates = value

    # ── alpha schedule ────────────────────────────────────────────────────────
    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def _alpha_current(self) -> float:
        if self._warmup_epochs <= 0:
            return self._alpha_final
        progress = min(self._current_epoch / self._warmup_epochs, 1.0)
        return self._alpha_warmup + progress * (self._alpha_final - self._alpha_warmup)

    # ── hook registration ─────────────────────────────────────────────────────
    def _register_feat_hooks(self):
        """
        Tap the last `feat_layers` backbone-neck output blocks from both student
        and teacher models.  We look for the Detect head's input tensors, which
        are the outputs of the final FPN neck stages (P3, P4, P5).
        """
        # We will defer actual hook registration to the first forward pass because
        # the student model may not yet be on the correct device.  Just store refs.
        pass

    def _maybe_init_hooks(self, student_model: nn.Module):
        """
        Called lazily on the first batch so all models are on device.
        Hooks last `feat_layers` conv blocks of each teacher + the student.
        """
        if self._hooks:
            return

        def _hook_factory(storage: list):
            def _hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    storage.append(out)
            return _hook

        def _tap_layers(model: nn.Module, storage: list, n: int):
            """Register hooks on the last `n` spatial-conv blocks of the model."""
            named = list(model.named_modules())
            candidates = [
                (name, m) for name, m in named
                if isinstance(m, (nn.Conv2d,))
                and "detect" not in name.lower()
                and "head" not in name.lower()
            ]
            if len(candidates) < n:
                to_tap = candidates
            else:
                step = max(len(candidates) // n, 1)
                to_tap = candidates[-n * step :: step][-n:]

            hooks = []
            for _, m in to_tap:
                h = m.register_forward_hook(_hook_factory(storage))
                hooks.append(h)
            return hooks

        all_hooks = []
        for teacher, storage in zip(self._teachers, self._t_feats_per_teacher):
            all_hooks += _tap_layers(teacher, storage, self._feat_layers)
        all_hooks += _tap_layers(student_model, self._s_feats, self._feat_layers)
        self._hooks = all_hooks

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ── channel adapters ──────────────────────────────────────────────────────
    def _build_adapters(self, all_teacher_feats, s_feats):
        """Build a 1×1 conv per (teacher, FPN level) to project teacher C → student C."""
        self._adapters = []
        for t_feats in all_teacher_feats:
            teacher_adapters = []
            for tf, sf in zip(t_feats, s_feats):
                tc, sc = tf.shape[1], sf.shape[1]
                if tc != sc:
                    adapter = _ChannelAdapter(tc, sc).to(self._device)
                    # No gradient — we only use adapted teacher feats as targets
                    for p in adapter.parameters():
                        p.requires_grad = False
                    teacher_adapters.append(adapter)
                else:
                    teacher_adapters.append(None)
            self._adapters.append(teacher_adapters)
        self._adapters_ready = True

    # ── NWD additive term for small GT boxes ─────────────────────────────────
    def _nwd_small_obj_loss(self, preds, batch) -> torch.Tensor:
        """
        Normalized Wasserstein Distance loss for GT boxes with √(w·h) < thresh.

        CIoU / DFL produce zero gradient when prediction and GT don't overlap,
        which is common for tiny boxes at early training.  NWD models boxes as
        2-D Gaussians so gradient flows even when boxes are far apart.

        Uses the decoded inference output (preds[0]: [B, 4+nc, N] xyxy pixels)
        to find the best-matching prediction for each small GT box, then applies
        NWD loss = 1 − NWD(pred, gt).  The matching step is detached; only the
        selected prediction has gradient, avoiding double-backward issues.
        """
        if self._nwd_weight <= 0:
            return torch.tensor(0.0, device=self._device)

        # preds[0] must be a decoded inference tensor [B, 4+nc, N] xyxy pixels
        if not isinstance(preds, tuple) or not isinstance(preds[0], torch.Tensor):
            return torch.tensor(0.0, device=self._device)

        decoded = preds[0]  # [B, 4+nc, N]
        if decoded.ndim != 3 or decoded.shape[1] < 4:
            return torch.tensor(0.0, device=self._device)

        img_size = float(batch["img"].shape[-1])

        # Decoded xyxy → xywh, normalized → pixel space
        # preds[0] channels: [x1, y1, x2, y2, cls0, cls1, ...]
        x1, y1 = decoded[:, 0, :], decoded[:, 1, :]
        x2, y2 = decoded[:, 2, :], decoded[:, 3, :]
        pred_cx = (x1 + x2) / 2   # [B, N]
        pred_cy = (y1 + y2) / 2
        pred_w  = (x2 - x1).clamp(min=0)
        pred_h  = (y2 - y1).clamp(min=0)
        # Stack: [B, N, 4] in pixel space (cx, cy, w, h)
        pred_xywh_px = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

        # GT boxes: normalized xywh → pixel space
        gt_bboxes = batch["bboxes"]   # [total_gt, 4]  normalized xywh
        gt_batch  = batch["batch_idx"].long()  # [total_gt]

        gt_area       = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=0).sqrt()
        small_mask    = gt_area < self._nwd_small_thresh
        if small_mask.sum() == 0:
            return torch.tensor(0.0, device=self._device)

        small_boxes   = gt_bboxes[small_mask] * img_size   # [K, 4] pixels
        small_batch   = gt_batch[small_mask]                # [K]

        nwd_losses: list[torch.Tensor] = []
        B = decoded.shape[0]
        for b_idx in range(B):
            img_small = (small_batch == b_idx)
            if img_small.sum() == 0:
                continue

            gt_px  = small_boxes[img_small]          # [k, 4] pixels, no grad
            pred_n = pred_xywh_px[b_idx]             # [N, 4] pixels, grad on

            with torch.no_grad():
                sim_detached = _nwd_similarity(pred_n.detach(), gt_px)  # [k, N]
                best_idx     = sim_detached.argmax(dim=1)               # [k]

            # Recompute NWD with gradient flowing only through selected predictions
            best_pred = pred_n[best_idx]             # [k, 4]  grad on
            sim_grad  = _nwd_similarity(best_pred, gt_px)  # [k, k] diagonal has real match
            # We want the diagonal: sim_grad[i, i] = NWD(best_pred_i, gt_i)
            nwd_sim   = sim_grad.diagonal()           # [k]
            nwd_losses.append((1.0 - nwd_sim).mean())

        if not nwd_losses:
            return torch.tensor(0.0, device=self._device)

        return torch.stack(nwd_losses).mean() * self._nwd_weight

    # ── main entry point ──────────────────────────────────────────────────────
    def __call__(self, preds, batch):
        det_loss, loss_items = self._base(preds, batch)

        B       = batch["img"].shape[0]
        alpha   = self._alpha_current()

        # ── Head KD ──────────────────────────────────────────────────────────
        head_kd = self._compute_head_kd(preds, batch["img"]) * B

        # ── Feature KD ───────────────────────────────────────────────────────
        if self._feat_beta > 0:
            feat_kd = self._compute_feat_kd() * B   # clears _s_feats internally
        else:
            self._s_feats.clear()   # still must clear so hooks don't accumulate
            feat_kd = torch.tensor(0.0, device=self._device)

        # ── NWD additive term (small objects, zero-grad fix) ─────────────────
        nwd_loss = self._nwd_small_obj_loss(preds, batch)

        # ── Combined loss ─────────────────────────────────────────────────────
        total = (
            (1.0 - alpha) * det_loss
            + alpha        * head_kd
            + self._feat_beta * feat_kd
            + nwd_loss
        )
        return total, loss_items

    # ── Response-based (head) KD — multi-teacher ensemble ─────────────────────
    def _compute_head_kd(self, student_preds, imgs: torch.Tensor) -> torch.Tensor:
        """
        KL divergence on class logits + DFL box distributions.
        Teachers are forwarded individually (each in its own no-grad context);
        their raw logits are averaged across the ensemble to produce the soft
        target.  Teacher features are captured here via forward hooks already
        registered on each teacher's backbone/neck.
        """
        T = self._T

        # Student raw outputs (already computed by trainer's forward pass)
        s = student_preds[1] if isinstance(student_preds, tuple) else student_preds

        # Clear per-teacher feature storage before teacher forwards
        # (student feats were captured during the student forward — keep them)
        for storage in self._t_feats_per_teacher:
            storage.clear()

        # ── Ensemble teacher forward pass ────────────────────────────────────
        t_scores_list: list[torch.Tensor] = []
        t_boxes_list:  list[torch.Tensor] = []
        with torch.no_grad():
            for teacher in self._teachers:
                t_out = teacher(imgs)
                if isinstance(t_out, tuple):
                    t_out = t_out[1]
                t_scores_list.append(t_out["scores"])
                t_boxes_list.append(t_out["boxes"])

        # Ensemble target = mean of teacher logits  (simple & robust)
        t_scores = torch.stack(t_scores_list, dim=0).mean(dim=0)   # [B, nc, N]
        t_boxes  = torch.stack(t_boxes_list,  dim=0).mean(dim=0)   # [B, 4*reg_max, N]

        # ── Classification KD ─────────────────────────────────────────────────
        s_scores = s["scores"]                             # [B, nc, N]
        B, nc, N = s_scores.shape
        s_cls = s_scores.permute(0, 2, 1).reshape(-1, nc)
        t_cls = t_scores.permute(0, 2, 1).reshape(-1, nc)

        kd_cls = F.kl_div(
            F.log_softmax(s_cls / T, dim=-1),
            F.softmax(t_cls / T, dim=-1).detach(),
            reduction="batchmean",
        ) * (T * T)

        # ── DFL box distribution KD ────────────────────────────────────────────
        s_boxes  = s["boxes"]                              # [B, 4*reg_max, N]
        reg_max  = s_boxes.shape[1] // 4
        s_box_flat = s_boxes.permute(0,2,1).reshape(-1, 4, reg_max).reshape(-1, reg_max)
        t_box_flat = t_boxes.permute(0,2,1).reshape(-1, 4, reg_max).reshape(-1, reg_max)

        kd_box = F.kl_div(
            F.log_softmax(s_box_flat / T, dim=-1),
            F.softmax(t_box_flat / T, dim=-1).detach(),
            reduction="batchmean",
        ) * (T * T)

        return kd_cls + 0.5 * kd_box

    # ── Feature AT KD — multi-teacher ensemble ────────────────────────────────
    def _compute_feat_kd(self) -> torch.Tensor:
        """
        Attention-transfer L2 loss between student and mean-of-teachers
        attention maps, summed over FPN levels.
        """
        s_feats = self._s_feats
        all_teacher_feats = self._t_feats_per_teacher

        if not s_feats or not all_teacher_feats or not all(all_teacher_feats):
            return torch.tensor(0.0, device=self._device)

        # Lazy-init per-(teacher, level) adapters
        if not self._adapters_ready:
            self._build_adapters(all_teacher_feats, s_feats)

        n_levels = min(
            len(s_feats),
            min(len(tf) for tf in all_teacher_feats),
        )
        loss = torch.tensor(0.0, device=self._device)

        for lvl in range(n_levels):
            sf   = s_feats[lvl]
            at_s = _attention_map(sf)   # [B, 1, H*W]

            # Average teacher attention maps at this level
            at_t_maps = []
            for t_idx, t_feat_list in enumerate(all_teacher_feats):
                tf = t_feat_list[lvl].detach()

                adapter = self._adapters[t_idx][lvl]
                if adapter is not None:
                    with torch.no_grad():
                        tf = adapter(tf)

                if tf.shape[-2:] != sf.shape[-2:]:
                    tf = F.adaptive_avg_pool2d(tf, sf.shape[-2:])

                at_t_maps.append(_attention_map(tf))

            at_t_mean = torch.stack(at_t_maps, dim=0).mean(dim=0)  # [B,1,H*W]
            loss = loss + F.mse_loss(at_s, at_t_mean.detach())

        # Clear student feature buffer after use — hooks append each forward pass
        # so without this, stale freed-graph tensors accumulate across iterations
        # and cause "backward through the graph a second time" on iteration 2+.
        self._s_feats.clear()

        return loss / max(n_levels, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  KD Trainer factory
# ══════════════════════════════════════════════════════════════════════════════

def make_kd_trainer(
    teacher_paths,
    kd_temperature:   float = 6.0,
    kd_alpha:         float = 0.65,
    kd_alpha_warmup:  float = 0.0,
    kd_warmup_epochs: int   = 10,
    total_epochs:     int   = 100,
    kd_feat_beta:     float = 0.30,
    kd_feat_layers:   int   = 3,
    nwd_weight:       float = 1.5,
    nwd_small_thresh: float = 0.04,
    full_cfg:         dict  = None,
    dataset_yaml:     str   = None,
):
    """
    Returns a DetectionTrainer subclass with a frozen teacher **ensemble**
    baked in.

    `teacher_paths` may be a single path (str) or a list of paths. Multiple
    teachers are forwarded per batch and their logits averaged to form the
    soft target for head KD; their attention maps are averaged for feature KD.

    A factory is used because Ultralytics instantiates the trainer class
    internally (model.train(trainer=...)), so we cannot pass kwargs through the
    constructor. The closure captures all KD hyperparams.
    """
    if isinstance(teacher_paths, (str, Path)):
        teacher_paths = [str(teacher_paths)]
    _tp_list          = [str(p) for p in teacher_paths]
    _T                = float(kd_temperature)
    _alpha            = float(kd_alpha)
    _alpha_warmup     = float(kd_alpha_warmup)
    _warmup_epochs    = int(kd_warmup_epochs)
    _total_epochs     = int(total_epochs)
    _feat_beta        = float(kd_feat_beta)
    _feat_layers      = int(kd_feat_layers)
    _nwd_weight       = float(nwd_weight)
    _nwd_small_thresh = float(nwd_small_thresh)
    _full_cfg         = full_cfg   or {}
    _dataset_yaml     = dataset_yaml or ""

    class WeaponKDTrainer(DetectionTrainer):
        """
        YOLOv8n student trainer with:
          • Multi-teacher response-based KD (ensemble-averaged logits)
          • Multi-teacher FPN neck feature Attention Transfer KD
          • Adaptive alpha warmup scheduling
        """

        # ── Attach frozen teacher ensemble during model build ──────────────────
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            self._load_teachers()
            return model

        def _load_teachers(self):
            teachers = []
            for tp in _tp_list:
                teacher_yolo = YOLO(tp)
                tm           = teacher_yolo.model.to(self.device)

                for p in tm.parameters():
                    p.requires_grad = False

                # Keep BN in eval mode so the teacher uses its own running stats
                # rather than being corrupted by the student's batch stats.
                tm.train()
                for m in tm.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

                teachers.append(tm)

            self._teachers        = teachers
            self._kd_T            = _T
            self._kd_alpha        = _alpha
            self._kd_alpha_warmup = _alpha_warmup
            self._kd_warmup_ep    = _warmup_epochs
            self._total_ep        = _total_epochs
            self._feat_beta       = _feat_beta
            self._feat_layers     = _feat_layers

            print(f"\n{'─'*65}")
            print(f"  [KD] Teacher ensemble ({len(teachers)}):")
            for i, tp in enumerate(_tp_list):
                print(f"         [{i}] {tp}")
            print(f"  [KD] Temperature    : T = {_T}")
            print(f"  [KD] Alpha schedule : {_alpha_warmup:.2f} → {_alpha:.2f}  "
                  f"over {_warmup_epochs} warmup epochs")
            print(f"  [KD] Feat β (AT)    : {_feat_beta}  ({_feat_layers} FPN levels)")
            print(f"{'─'*65}\n")

        # ── Wrap criterion after parent setup ─────────────────────────────────
        def _setup_train(self):
            super()._setup_train()

            model = unwrap_model(self.model)

            # Force criterion init if Ultralytics hasn't done it yet
            if getattr(model, "criterion", None) is None:
                model.criterion = model.init_criterion()

            self._kd_criterion = _KDCriterion(
                base_criterion   = model.criterion,
                teachers         = self._teachers,
                T                = self._kd_T,
                alpha            = self._kd_alpha,
                alpha_warmup     = self._kd_alpha_warmup,
                warmup_epochs    = self._kd_warmup_ep,
                total_epochs     = self._total_ep,
                feat_beta        = self._feat_beta,
                feat_layers      = self._feat_layers,
                device           = self.device,
                nwd_weight       = _nwd_weight,
                nwd_small_thresh = _nwd_small_thresh,
            )
            model.criterion = self._kd_criterion

            # Lazy-init feature hooks now that both models are on device
            try:
                self._kd_criterion._maybe_init_hooks(model)
            except Exception as e:
                print(f"  [KD] feature-hook init failed ({e}) — feat KD disabled")

            # ── Small-object loss-guided injector ─────────────────────────
            self._small_obj_injector = None
            try:
                self._small_obj_injector = build_injector_from_cfg(
                    _full_cfg, _dataset_yaml
                )
                if self._small_obj_injector is not None:
                    print("  [SmallObjAug] Loss-guided injector attached")
            except Exception as e:
                print(f"  [SmallObjAug] init failed ({e}) — disabled")

        # ── Override preprocess_batch to inject small-object samples ──────
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            inj = getattr(self, "_small_obj_injector", None)
            if inj is not None:
                img_size = batch["img"].shape[-1]
                batch = inj.process_batch(batch, img_size, self.device)
            return batch

        # ── Update KD alpha each optimizer step (epoch tracked via self.epoch) ─
        def optimizer_step(self):
            # Forward epoch to KD criterion for alpha scheduling
            if hasattr(self, "_kd_criterion"):
                self._kd_criterion.set_epoch(self.epoch)
            super().optimizer_step()

    return WeaponKDTrainer


# ══════════════════════════════════════════════════════════════════════════════
#  Public training entry point
# ══════════════════════════════════════════════════════════════════════════════

def train_student(cfg, teacher_ckpts="logs/detector/best.pt"):
    """
    Train a YOLOv8n student at img_size×img_size with three-term KD from
    one or more trained teachers (response + feature AT + adaptive alpha).

    Args:
        cfg:           Full config dict (from hyperparams.yaml).
        teacher_ckpts: Path (str) or list of paths to teacher best.pt files.
                       Each must be a YOLOv8 detector with the same nc as the
                       student (nc=3 here). A list enables multi-teacher
                       ensemble distillation (logits and feature attention are
                       averaged across teachers).

    Returns:
        str | None: Path to student best.pt, or None on failure.
    """
    # ── Normalise teacher input to a list and filter missing files ───────────
    if isinstance(teacher_ckpts, (str, Path)):
        teacher_ckpts = [str(teacher_ckpts)]
    else:
        teacher_ckpts = [str(t) for t in teacher_ckpts]

    # Dedupe while preserving order
    seen = set()
    teacher_ckpts = [
        t for t in teacher_ckpts
        if not (t in seen or seen.add(t))
    ]

    existing = [t for t in teacher_ckpts if Path(t).exists()]
    missing  = [t for t in teacher_ckpts if not Path(t).exists()]
    for m in missing:
        print(f"  [WARN] Teacher weights not found (skipping): {m}")

    if not existing:
        print(f"  [SKIP] No valid teacher weights provided — aborting student training")
        return None

    print(f"  [KD] Using {len(existing)} teacher(s) for distillation:")
    for i, tp in enumerate(existing):
        print(f"         [{i}] {tp}")

    st_cfg = cfg.get("student", {})
    aug    = cfg.get("augmentation", {})
    aerial = aug.get("aerial", {})

    save_dir = Path(cfg["logging"]["save_dir"]) / "student"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name       = st_cfg.get("model",            "yolov8n.pt")
    img_size         = st_cfg.get("img_size",          416)
    epochs           = st_cfg.get("epochs",            100)
    batch            = st_cfg.get("batch_size",         32)
    lr0              = st_cfg.get("lr0",               0.01)
    lrf              = st_cfg.get("lrf",               0.001)
    workers          = st_cfg.get("workers",             4)
    kd_temperature   = st_cfg.get("kd_temperature",    6.0)
    kd_alpha         = st_cfg.get("kd_alpha",          0.65)
    kd_alpha_warmup  = st_cfg.get("kd_alpha_warmup",   0.0)
    kd_warmup_epochs = st_cfg.get("kd_warmup_epochs",  10)
    kd_feat_beta     = st_cfg.get("kd_feat_beta",       0.30)
    kd_feat_layers   = st_cfg.get("kd_feat_layers",    3)
    nwd_weight       = st_cfg.get("nwd_weight",        1.5)
    nwd_small_thresh = st_cfg.get("nwd_small_thresh",  0.04)
    label_smoothing  = st_cfg.get("label_smoothing",   0.05)
    patience         = st_cfg.get("patience",          25)
    save_period      = st_cfg.get("save_period",       10)

    dataset_yaml = str(Path(cfg["dataset"]["root"]) / "dataset.yaml")

    # ── Determine initial student weights ─────────────────────────────────────
    # Priority (first match wins):
    #   1. Resume interrupted run: logs/student/weights/last.pt exists AND
    #      no completed-run artifact (logs/student/best.pt) exists newer than it
    #   2. student.init_from — explicit path override from config
    #   3. logs/student/best.pt — previous completed distillation (iterative KD)
    #   4. student.model — fresh pretrain (e.g. yolo11n.pt COCO weights)
    init_override = st_cfg.get("init_from", None)
    prev_best     = save_dir / "best.pt"                 # our final-artifact copy
    last_ckpt     = save_dir / "weights" / "last.pt"     # Ultralytics per-epoch save

    # Heuristic: a run was interrupted if last.pt exists AND either
    #   (a) no final artifact copy exists (prev_best missing), OR
    #   (b) last.pt is newer than prev_best (restart was after completion)
    want_resume = last_ckpt.exists() and (
        not prev_best.exists()
        or last_ckpt.stat().st_mtime > prev_best.stat().st_mtime
    )

    do_resume = False
    if want_resume:
        init_weights = str(last_ckpt)
        init_source  = "resume interrupted run (last.pt)"
        do_resume    = True
    elif init_override and Path(init_override).exists():
        init_weights = str(init_override)
        init_source  = "config.init_from"
    elif prev_best.exists():
        # Snapshot the previous student so the training run can't clobber our
        # warm-start weights mid-training.
        init_snapshot = save_dir / "init_prev_student.pt"
        shutil.copy2(str(prev_best), str(init_snapshot))
        init_weights = str(init_snapshot)
        init_source  = "previous student (iterative KD)"
    else:
        init_weights = model_name
        init_source  = "fresh pretrain"

    print(f"\n{'═'*65}")
    print(f"  Step 9 — Student Distillation (multi-teacher KD + Feature AT)")
    print(f"  Teachers      : {len(existing)}")
    for i, tp in enumerate(existing):
        print(f"                  [{i}] {tp}")
    print(f"  Student arch  : {model_name}  →  {img_size}×{img_size}px")
    print(f"  Init weights  : {init_weights}  [{init_source}]")
    print(f"  KD T={kd_temperature}  α={kd_alpha_warmup}→{kd_alpha}  "
          f"β={kd_feat_beta}  warmup={kd_warmup_epochs}ep")
    print(f"  Epochs        : {epochs}   batch={batch}   lr0={lr0}")
    print(f"  Label smooth  : {label_smoothing}")
    print(f"  Early stop    : patience={patience} epochs (monitors best fitness)")
    print(f"  Save period   : every {save_period} epochs → weights/epoch*.pt")
    print(f"  Resume        : {'YES — continuing from last.pt' if do_resume else 'NO (fresh start)'}")
    print(f"  Save dir      : {save_dir}")
    print(f"{'═'*65}\n")

    # Build the KD trainer
    KDTrainer = make_kd_trainer(
        teacher_paths    = existing,
        kd_temperature   = kd_temperature,
        kd_alpha         = kd_alpha,
        kd_alpha_warmup  = kd_alpha_warmup,
        kd_warmup_epochs = kd_warmup_epochs,
        total_epochs     = epochs,
        kd_feat_beta     = kd_feat_beta,
        kd_feat_layers   = kd_feat_layers,
        nwd_weight       = nwd_weight,
        nwd_small_thresh = nwd_small_thresh,
        full_cfg         = cfg,
        dataset_yaml     = dataset_yaml,
    )

    # Student model — load warm-start weights with graceful fallback to arch
    try:
        model = YOLO(init_weights)
        if init_source != "fresh pretrain":
            print(f"  [Student] Warm-started from {init_weights}")
    except Exception as e:
        print(f"  [WARN] Failed to load {init_weights} ({e})")
        print(f"  [WARN] Falling back to fresh pretrain: {model_name}")
        model = YOLO(model_name)
        init_source  = "fresh pretrain (fallback)"
        init_weights = model_name

    # ── Monitor callbacks ──────────────────────────────────────────────────────
    mon = TrainingMonitor(log_dir=str(save_dir), step_name="student")
    mon.log_event(
        f"Start KD: {len(existing)} teachers, T={kd_temperature} "
        f"α={kd_alpha_warmup}→{kd_alpha} β={kd_feat_beta} "
        f"{epochs}ep {img_size}px bs={batch} patience={patience}"
    )

    def _on_epoch_start(trainer):
        mon.start_epoch(trainer.epoch, trainer.epochs)

    def _on_epoch_end(trainer):
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k.split("/")[-1]] = round(float(v), 4)
        if hasattr(trainer, "loss") and trainer.loss is not None:
            try:
                metrics["loss"] = round(float(trainer.loss), 4)
            except Exception:
                pass
        mon.end_epoch(metrics)

        # Per-class mAP50 log so rifle regression is visible immediately
        if hasattr(trainer, "metrics") and trainer.metrics:
            m = trainer.metrics
            class_names = ["knife", "pistol", "rifle"]
            per_cls = []
            for i, name in enumerate(class_names):
                key = f"metrics/mAP50(B)/{name}"   # Ultralytics 8.x key
                if key in m:
                    per_cls.append(f"{name}={float(m[key]):.3f}")
            if per_cls:
                mon.log_event("  per-class mAP50: " + "  ".join(per_cls))

    # ── Early-stopping telemetry (Ultralytics already enforces `patience`) ──
    es_state = {"best_epoch": 0, "best_fitness": float("-inf")}

    def _on_fit_epoch_end(trainer):
        stopper = getattr(trainer, "stopper", None)
        if stopper is None:
            return
        be = getattr(stopper, "best_epoch", None)
        bf = getattr(stopper, "best_fitness", None)
        if be is None:
            return

        if bf is not None and float(bf) > es_state["best_fitness"]:
            es_state["best_fitness"] = float(bf)
            es_state["best_epoch"]   = int(be)
            mon.log_event(
                f"★ new best fitness={float(bf):.4f} @ epoch {int(be)}"
            )
            return

        stagnant = int(trainer.epoch) - int(be)
        remaining = max(0, int(patience) - stagnant)
        if stagnant > 0:
            mon.log_event(
                f"early-stop watch: {stagnant}/{patience} epochs since "
                f"best (epoch {int(be)}), {remaining} before stop"
            )

    model.add_callback("on_train_epoch_start", _on_epoch_start)
    model.add_callback("on_train_epoch_end",   _on_epoch_end)
    model.add_callback("on_fit_epoch_end",     _on_fit_epoch_end)

    # ── Launch training ────────────────────────────────────────────────────────
    results = model.train(
        trainer          = KDTrainer,
        data             = dataset_yaml,
        epochs           = epochs,
        imgsz            = img_size,
        batch            = batch,
        lr0              = lr0,
        lrf              = lrf,
        momentum         = cfg["detector"].get("momentum",      0.937),
        weight_decay     = cfg["detector"].get("weight_decay",  0.0005),
        warmup_epochs    = cfg["detector"].get("warmup_epochs", 5),
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,
        patience         = patience,
        save_period      = save_period,
        resume           = do_resume,
        project          = str(cfg["logging"]["save_dir"]),
        name             = "student",
        exist_ok         = True,
        device           = 0,
        workers          = workers,
        amp              = cfg["hardware"]["amp"],
        cos_lr           = True,
        close_mosaic     = 15,
        label_smoothing  = label_smoothing,

        # ── Augmentation (more aggressive for student to compensate lower capacity)
        hsv_h     = aug.get("hsv_h",     0.05),
        hsv_s     = aug.get("hsv_s",     0.7),
        hsv_v     = aug.get("hsv_v",     0.4),
        fliplr    = aug.get("fliplr",    0.5),
        flipud    = aug.get("flipud",    0.5),
        scale     = aug.get("scale",     0.6),
        translate = aug.get("translate", 0.2),
        mosaic    = aug.get("mosaic",    1.0),
        mixup     = aug.get("mixup",     0.2),
        copy_paste= 0.1,

        # Aerial augmentations
        degrees     = aerial.get("degrees",      15.0),
        perspective = aerial.get("perspective",  0.0005),
        shear       = aerial.get("shear",        3.0),

        # Loss weights — slightly upweight box/cls for a 3-class detector
        box       = 7.5,
        cls       = 0.5,
        dfl       = 1.5,

        verbose = True,
        plots   = True,
    )

    mon.log_event("Student KD training complete")
    mon.close()

    # ── Locate and copy best weights ──────────────────────────────────────────
    possible = [
        save_dir / "weights" / "best.pt",
        Path(cfg["logging"]["save_dir"]) / "student" / "weights" / "best.pt",
    ]
    yolo_best = next((p for p in possible if p.exists()), None)
    our_best  = save_dir / "best.pt"

    if yolo_best:
        shutil.copy2(str(yolo_best), str(our_best))
        print(f"\n  [Student] Saved → {our_best}")
    else:
        print(f"\n  [WARN] Student best.pt not found — check {save_dir}")
        return None

    # ── Final metrics ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Student KD Training Complete")
    try:
        m = results.results_dict
        print(f"  mAP50    : {m.get('metrics/mAP50(B)',     0):.4f}")
        print(f"  mAP50-95 : {m.get('metrics/mAP50-95(B)',  0):.4f}")
        print(f"  Precision: {m.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall   : {m.get('metrics/recall(B)',    0):.4f}")
    except Exception:
        pass
    print(f"  Weights  : {our_best}")
    print(f"{'═'*65}")

    return str(our_best)


# ══════════════════════════════════════════════════════════════════════════════
#  Student export
# ══════════════════════════════════════════════════════════════════════════════

def export_student(cfg, student_ckpt: str = "logs/student/best.pt"):
    """
    Export the student model to NCNN (Pi 5 CPU) and ONNX (Hailo HEF pipeline).

    Args:
        cfg:          Full config dict.
        student_ckpt: Path to student best.pt.

    Returns:
        dict: {'ncnn_param': str, 'ncnn_bin': str, 'onnx': str}
    """
    if not Path(student_ckpt).exists():
        print(f"  [SKIP] Student weights not found: {student_ckpt}")
        return {}

    st_cfg     = cfg.get("student", {})
    img_size   = st_cfg.get("img_size", 416)
    export_dir = Path(cfg["logging"]["save_dir"]) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    model   = YOLO(student_ckpt)
    exports = {}

    print(f"\n  Student export ({img_size}px) for Pi 5 AI HAT:")
    print(f"    Source : {student_ckpt}")

    # ── NCNN export — Pi 5 CPU fallback ───────────────────────────────────────
    try:
        t0        = time.time()
        ncnn_path = model.export(
            format   = "ncnn",
            imgsz    = img_size,
            simplify = True,
        )
        elapsed  = time.time() - t0
        ncnn_dir = Path(str(ncnn_path))
        if ncnn_dir.is_dir():
            for f in ncnn_dir.iterdir():
                dst = export_dir / f"student_{f.name}"
                shutil.copy2(f, dst)
                if f.suffix == ".param":
                    exports["ncnn_param"] = str(dst)
                elif f.suffix == ".bin":
                    exports["ncnn_bin"] = str(dst)
        print(f"    NCNN   : {export_dir}/student_*.{{param,bin}}  ({elapsed:.1f}s)")
        print(f"    → Copy to Pi 5 and run with ncnn's yolov8_demo")
    except Exception as e:
        print(f"    NCNN   : FAILED — {e}")

    # ── ONNX export — intermediate for Hailo HEF ──────────────────────────────
    try:
        t0        = time.time()
        onnx_path = model.export(
            format   = "onnx",
            imgsz    = img_size,
            opset    = 11,
            simplify = True,
            dynamic  = False,
        )
        elapsed = time.time() - t0
        dst = export_dir / "student_detector.onnx"
        if onnx_path and Path(str(onnx_path)).exists():
            shutil.copy2(str(onnx_path), str(dst))
            size_mb = dst.stat().st_size / 1e6
            exports["onnx"] = str(dst)
        print(f"    ONNX   : {dst}  ({size_mb:.1f} MB, {elapsed:.1f}s)")
        print(f"    → Convert to Hailo HEF with:")
        print(f"      hailomz compile --hw-arch hailo8l \\")
        print(f"        --onnx {dst} \\")
        print(f"        --calib-path data/images/val/ \\")
        print(f"        --classes 3 \\")
        print(f"        -o {export_dir}/student_detector.hef")
    except Exception as e:
        print(f"    ONNX   : FAILED — {e}")

    return exports
