import yaml
import torch
import logging

# Load YAML configuration file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize logging
def setup_logging(config):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(config['log_path'], mode='a', encoding='utf-8'), 
                            logging.StreamHandler()
                        ])
    logging.info("Training started with the following hyperparameters:")
    for key, value in config.items():
        logging.info(f'{key}: {value}')

# Create optimizers and learning rate schedulers
def create_optimizer_scheduler(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    lr_scheduler_type = config['lr_scheduler']
    gamma = config['gamma']
    plateau_patience = config['plateau_patience']
    plateau_threshold = config['plateau_threshold']

    if lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=plateau_threshold)
    elif lr_scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    elif lr_scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    return optimizer, scheduler

def get_device(config):
    return torch.device("cuda" if torch.cuda.is_available() and not config.get('disable_cuda', False) else "cpu")

# Load all configuration parameters and return the configuration in dictionary form
def parse_arguments():
    config = load_config('config_vit.yaml')
    return config
