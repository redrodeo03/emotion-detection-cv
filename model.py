import torch.nn as nn
from transformers import VivitModel, VivitConfig

class ViViTForClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vivit = VivitModel(VivitConfig())
        self.classifier = nn.Linear(self.vivit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vivit(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits