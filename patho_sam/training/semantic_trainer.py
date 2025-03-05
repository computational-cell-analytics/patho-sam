from typing import Optional, List

import torch
import torch.nn as nn

from micro_sam.training import SemanticSamTrainer
from micro_sam.training.semantic_sam_trainer import CustomDiceLoss


class SemanticInstanceTrainer(SemanticSamTrainer):
    """Modified trainer class for training the Segment Anything Model for semantic (instance) segmentation.
    """
    def __init__(
        self,
        convert_inputs,
        num_classes: int,
        dice_weight: Optional[float] = None,
        class_weights: Optional[List[float]] = None,
        **kwargs
    ):
        assert num_classes > 1

        if "loss" not in kwargs:
            kwargs["loss"] = CustomDiceLoss(num_classes=num_classes)

        if "metric" not in kwargs:
            kwargs["metric"] = CustomDiceLoss(num_classes=num_classes)

        super().__init__(convert_inputs=convert_inputs, num_classes=num_classes, dice_weight=dice_weight, **kwargs)

        self.class_weights = class_weights
        if self.class_weights is None:
            self.compute_ce_loss = nn.CrossEntropyLoss()
        else:
            weight_vals = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
            self.compute_ce_loss = nn.CrossEntropyLoss(weight=weight_vals)

        if self.dice_weight is not None and (self.dice_weight < 0 or self.dice_weight > 1):
            raise ValueError("The weight factor should lie between 0 and 1.")

        self._kwargs = kwargs

    def _get_model_outputs(self, batched_inputs):
        """Get the predictions from the model.
        """
        inputs = torch.stack([bi["image"] for bi in batched_inputs], dim=0).to(self.device)
        outputs = self.model(inputs.to(self.device))
        return outputs
