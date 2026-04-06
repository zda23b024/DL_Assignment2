"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.

        Args:
            eps: Small value to avoid division by zero.
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] (x_center, y_center, width, height)
            target_boxes: [B, 4] (x_center, y_center, width, height)

        Returns:
            IoU loss
        """

        # Convert (x_center, y_center, w, h) → (x1, y1, x2, y2)
        def to_corners(boxes):
            x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            x1 = x_c - w / 2.0
            y1 = y_c - h / 2.0
            x2 = x_c + w / 2.0
            y2 = y_c + h / 2.0

            return x1, y1, x2, y2

        pred_x1, pred_y1, pred_x2, pred_y2 = to_corners(pred_boxes)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = to_corners(target_boxes)

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)

        intersection = inter_w * inter_h

        # Areas
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)

        # Union
        union = pred_area + tgt_area - intersection + self.eps

        # IoU
        iou = intersection / union

        # Loss = 1 - IoU (range [0,1])
        loss = 1.0 - iou

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss