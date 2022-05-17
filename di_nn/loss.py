from torch.nn import Module, L1Loss
from torch.nn import Module


class CustomL1Loss(Module):
    def __init__(self, is_metadata_aware=True):
        super().__init__()

        self.loss = L1Loss(reduction="none")
        self.target_key = "source_coordinates"
        self.is_metadata_aware = is_metadata_aware

    def forward(self, model_output, targets, mean_reduce=True):
        targets = targets[self.target_key]
        if model_output.shape != targets.shape:
            raise ValueError(
                "Model output's shape is {}, target's is {}".format(
                    model_output.shape, targets.shape
            ))
        loss = self.loss(model_output, targets)
        if mean_reduce:
            loss = loss.mean()
        
        return loss


LOSS_NAME_TO_CLASS_MAP = {
    "l1": CustomL1Loss,
}
