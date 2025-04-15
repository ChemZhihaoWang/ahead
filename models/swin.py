import torch.nn as nn
from transformers import SwinModel 

class CustomSwinTransformer(nn.Module):
    def __init__(self, swin_model, dropout_rate):
        super(CustomSwinTransformer, self).__init__()
        self.swin = swin_model  
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.swin.config.hidden_size, 1)

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        # Get hidden states from SwinModel
        outputs = self.swin(x, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        # Get CLS token features
        x = outputs.last_hidden_state[:, 0, :]  
        
        x = self.dropout(x)
        x = self.classifier(x)

        if output_attentions:
            attentions = outputs.attentions  
            return x, attentions
        
        return x
