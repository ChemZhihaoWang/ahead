import torch
import torch.nn as nn

class ImageGatingNetwork(nn.Module):
    """
    Lightweight visual gating network:
    - Small CNN to preserve spatial inductive bias
    - Global average pooling to 1x1
    - Linear -> Softmax to produce expert weights
    """
    def __init__(self, num_experts: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(256, num_experts)
        # Learnable temperature (softplus keeps it positive, add floor for stability)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.min_tau = 0.3
        # Learnable smoothing factor (sigmoid keeps it in [0,1])
        self.logit_smooth = nn.Parameter(torch.tensor(-2.1972))  # ~0.1 initial smoothing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W (same as experts' input)
        feat = self.features(x)                  # B x 128 x H' x W'
        gap = feat.mean(dim=(2, 3))              # B x 128
        gmp = feat.amax(dim=(2, 3))              # B x 128
        h = torch.cat([gap, gmp], dim=1)         # B x 256
        h = self.norm(h)
        h = self.dropout(h)
        logits = self.fc(h)                       # B x num_experts

        tau = torch.nn.functional.softplus(self.log_tau) + self.min_tau
        weights = torch.softmax(logits / tau, dim=1)

        smooth = torch.sigmoid(self.logit_smooth)
        if smooth.item() > 0:
            uniform = torch.ones_like(weights) / weights.size(1)
            weights = (1 - smooth) * weights + smooth * uniform
        return weights

class MoEModel(nn.Module):
    def __init__(self, experts, input_dim):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        # Use lightweight visual gating; keep input_dim for backward compatibility (unused)
        self.gating_network = ImageGatingNetwork(num_experts=self.num_experts)

    def forward(self, x):

        gate_weights = self.gating_network(x) 

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 

        gate_weights = gate_weights.unsqueeze(-1) 
        final_output = torch.sum(gate_weights * expert_outputs, dim=1) 
        return final_output

from config import args
from torchvision.models import resnet18, resnet34, resnet152, ResNet18_Weights, ResNet34_Weights, ResNet152_Weights
from models.resnet import CustomResNetModel
from models.vit import ViTModel, CustomViTModel
from models.swin import CustomSwinTransformer
from transformers import SwinModel

config = args.parse_arguments()

def create_expert_models(device):
    experts = []


    experts.append(CustomResNetModel(resnet18(weights=ResNet18_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device))

    # experts.append(CustomResNetModel(resnet34(weights=ResNet34_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device))

    # experts.append(CustomResNetModel(resnet152(weights=ResNet152_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device))

    vit_base_1 = ViTModel.from_pretrained(r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch16-224')
    experts.append(CustomViTModel(vit_base_1, dropout_rate=config['dropout_rate']).to(device))

    # vit_base_2 = ViTModel.from_pretrained(r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch32-224-in21k')
    # experts.append(CustomViTModel(vit_base_2, dropout_rate=config['dropout_rate']).to(device))

    # vit_large = ViTModel.from_pretrained(r'/home/wangzh/.cache/huggingface/hub/models--google--vit-large-patch16-224')
    # experts.append(CustomViTModel(vit_large, dropout_rate=config['dropout_rate']).to(device))


    swin_small_model = SwinModel.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-small-patch4-window7-224')
    swin_small = CustomSwinTransformer(swin_model=swin_small_model, dropout_rate=config['dropout_rate']).to(device)
    experts.append(swin_small)

    # swin_base_model = SwinModel.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-base-patch4-window7-224')
    # swin_base = CustomSwinTransformer(swin_model=swin_base_model, dropout_rate=config['dropout_rate']).to(device)
    # experts.append(swin_base)

    return experts
