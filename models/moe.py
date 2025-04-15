import torch
import torch.nn as nn
from transformers import BertModel

class TransformerGatingNetwork(nn.Module):
    def __init__(self, num_experts, input_dim, embedding_dim=768):
        super(TransformerGatingNetwork, self).__init__()
        self.transformer = BertModel.from_pretrained(r'/home/wangzh/.cache/huggingface/bert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, num_experts),  # Mapping from Transformer output to experts
            nn.Softmax(dim=1)
        )
        self.input_projection = nn.Linear(input_dim, embedding_dim)  # Map inputs to Transformer's embedding dimensions

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.input_projection(x) 
        x = x.unsqueeze(1)  
        
        # Transformer Processing Sequence
        transformer_output = self.transformer(inputs_embeds=x).last_hidden_state 
        
        # Use the output of the first position to generate weights
        gating_weights = self.fc(transformer_output[:, 0, :]) 
        return gating_weights

class MoEModel(nn.Module):
    def __init__(self, experts, input_dim):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.gating_network = TransformerGatingNetwork(num_experts=self.num_experts, input_dim=input_dim)

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
