from scripts.base_trainer import BaseModelTrainer
from transformers import SwinModel  
from models.swin import CustomSwinTransformer
import torch
from utils.atten_rollout import attention_rollout

class SwinTrainer(BaseModelTrainer):
    def __init__(self, criterion, optimizer, scheduler, device, config, log_path, model=None):
        """
        Initialize the Swin Transformer model and set up the trainer.
        """
        # Load the pretrained Swin Transformer model based on the specified configuration in config
        if model is None:
            if config['pretrained_model_swin'] == 'swin_base':

                swin_model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')
            elif config['pretrained_model_swin'] == 'swin_large':

                swin_model = SwinModel.from_pretrained('microsoft/swin-large-patch4-window7-224')
            else:
                raise ValueError(f"Unsupported pretrained model: {config['pretrained_model_swin']}")

            # Use CustomSwinTransformer to add a custom classifier and dropout
            model = CustomSwinTransformer(swin_model=swin_model, dropout_rate=config['dropout_rate']).to(device)
        
        # Initialize the BaseModelTrainer with the model, criterion, optimizer, scheduler, and device
        super().__init__(model, criterion, optimizer, scheduler, device, config, log_path)

    def evaluate_attention_rollout(self, x):
        """
        Perform attention rollout for interpretability.
        """
        # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Get model outputs and attention weights
            outputs = self.model.swin(x, output_attentions=True)
            attentions = outputs.attentions
            # Use Attention Rollout to analyze attention
            rollout_attention = attention_rollout(attentions)

        return rollout_attention
