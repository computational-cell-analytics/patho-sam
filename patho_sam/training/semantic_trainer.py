import torch

from micro_sam.training import SemanticSamTrainer 


class SemanticInstanceTrainer(SemanticSamTrainer):
    """Modified trainer class for training the Segment Anything Model for semantic (instance) segmentation.
    """
    def _get_model_outputs(self, batched_inputs):
        """Get the predictions from the model.
        """
        inputs = torch.stack([bi["image"] for bi in batched_inputs], dim=0).to(self.device)
        outputs = self.model(inputs.to(self.device))
        return outputs
