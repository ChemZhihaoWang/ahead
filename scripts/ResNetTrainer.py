from scripts.base_trainer import BaseModelTrainer

class ResNetTrainer(BaseModelTrainer):
    def __init__(self, criterion, optimizer, scheduler, device, config, log_path, model):
        super().__init__(model, criterion, optimizer, scheduler, device, config, log_path)
