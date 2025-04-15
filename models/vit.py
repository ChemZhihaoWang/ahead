import torch.nn as nn
from transformers import ViTModel

class CustomViTModel(nn.Module):
    def __init__(self, vit_model, dropout_rate):
        super(CustomViTModel, self).__init__()
        self.vit = vit_model
        self.dropout = nn.Dropout(p=dropout_rate)

        # Use ViTModel's hidden_size
        self.classifier = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, x, output_attentions=False):
        # Get last_hidden_state from ViTModel
        outputs = self.vit(x, output_attentions=output_attentions)
        x = outputs.last_hidden_state[:, 0, :]  

        x = self.dropout(x)
        x = self.classifier(x)

        if output_attentions:
            attentions = outputs.attentions  
            return x, attentions
        
        return x
