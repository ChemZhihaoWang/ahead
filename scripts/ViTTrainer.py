from scripts.base_trainer import BaseModelTrainer
from transformers import ViTForImageClassification
from models.vit import CustomViTModel
import torch
from utils.atten_rollout import attention_rollout

class ViTTrainer(BaseModelTrainer):
    def __init__(self, criterion, optimizer, scheduler, device, config, log_path, model=None):
        """
        Initialize the ViT model and set up the trainer.
        """
        # Load the pretrained ViT model based on the specified configuration in config
        if model is None:
            if config['pretrained_model_vit'] == 'vit_base':
                vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            elif config['pretrained_model_vit'] == 'vit_large':
                vit_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
            else:
                raise ValueError(f"Unsupported pretrained model: {config['pretrained_model_vit']}")

            # Use CustomViTModel to add a custom classifier and dropout
            model = CustomViTModel(vit_model=vit_model, dropout_rate=config['dropout_rate']).to(device)
        
        # Initialize the BaseModelTrainer with the model, criterion, optimizer, scheduler, and device
        super().__init__(model, criterion, optimizer, scheduler, device, config, log_path)

    def evaluate_attention_rollout(self, x):

        self.model.eval()
        with torch.no_grad():
            outputs, attentions = self.model(x, output_attentions=True)
            rollout_attention = attention_rollout(attentions)

        return rollout_attention