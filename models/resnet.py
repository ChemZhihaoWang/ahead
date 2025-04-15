import torch

class CustomResNetModel(torch.nn.Module):
    def __init__(self, pretrained_model, dropout_rate):
        super().__init__()
        self.features = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
        self.dropout = torch.nn.Dropout(dropout_rate)
        num_ftrs = pretrained_model.fc.in_features
        self.fc = torch.nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    



