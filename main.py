import sys
import os
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)
from utils import utils
from dataset import dataloader
from models import resnet, vit, gin, moe 
from models.moe import create_expert_models
from models.swin import CustomSwinTransformer  
from scripts import test, predict
from scripts.ResNetTrainer import ResNetTrainer
from scripts.ViTTrainer import ViTTrainer
from scripts.GinTrainer import GinTrainer
from scripts.MoETrainer import MoETrainer
from scripts.SwinTrainer import SwinTrainer
from transformers import SwinModel
from config import args
from transformers import ViTModel
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# Set environment variables to ensure CuBLAS determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def configure_optimizers_schedulers(experts, config):
    optimizers = []
    schedulers = []

    for idx, model in enumerate(experts):
        lr = config['lr']
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(config['weight_decay']))
        scheduler = utils.configure_scheduler(optimizer, config)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    return optimizers, schedulers

def main():
    # Load all configuration parameters
    config = args.parse_arguments()

    if torch.cuda.is_available() and not config['disable_cuda']:
        gpu_id = config.get('gpu_id', config['gpu_id'])
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set up logging
    args.setup_logging(config)
    utils.set_seed(3407)

    all_train_actuals = []
    all_train_predictions = []
    all_test_actuals = []
    all_test_predictions = []

    if config['model'] in ['resnet', 'vit', 'swin']:
        checkpoint_dir_base = f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/checkpoints/{config['pretrained_model']}"
        os.makedirs(checkpoint_dir_base, exist_ok=True)
    else:
        checkpoint_dir_base = f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/checkpoints"
        os.makedirs(checkpoint_dir_base, exist_ok=True)

    # Initialize the list to store the mean and standard deviation of each fold
    mean_list = []
    std_list = []
    label_mean_list = []
    label_std_list = []

    if config['model'] == 'gin':
        hidden_dims = [
            config.get('hidden_dim_1'),
            config.get('hidden_dim_2'),
            config.get('hidden_dim_3'),
            config.get('hidden_dim_4'),
            config.get('hidden_dim_5')
        ]
        dropout_rate = config.get('dropout_rate')

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    # Load the full dataset index
    full_dataset_indices = list(
        range(len(pd.read_csv(config['train_annotations']))))

    # Start K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset_indices)):
        print(f'----- Starting Fold {fold + 1}/{k_folds} -----')

        # Create data sets and transformations
        if config['model'] in ['resnet', 'vit', 'moe', 'swin']: 
            dataset, _, mean, std = dataloader.prepare_dataset(
                config['train_annotations'],
                config['train_images'],
                config['batch_size'],
                device,
                train_indices=train_idx,
                is_training=True
            )
            mean_list.append(mean)
            std_list.append(std)
            label_mean_list.append(dataset.label_mean)
            label_std_list.append(dataset.label_std)
        elif config['model'] == 'gin':
            graph_dataset, _, mean, std, label_mean, label_std = dataloader.prepare_dataset_gin(
                config['train_annotations'],
                config['train_images'],
                config['batch_size'],
                device,
                train_indices=train_idx,
                is_training=True
            )
            mean_list.append(mean)
            std_list.append(std)
            label_mean_list.append(label_mean)
            label_std_list.append(label_std)
        else:
            raise ValueError("Unsupported model type")

        # Create training and validation datasets based on model type
        if config['model'] == 'gin':
            # Create training and validation subsets
            train_subset = [graph_dataset[i] for i in train_idx]
            val_subset = [graph_dataset[i] for i in val_idx]
            # Create data loaders
            from torch_geometric.loader import DataLoader as GeoDataLoader
            train_loader = GeoDataLoader(
                train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = GeoDataLoader(
                val_subset, batch_size=config['batch_size'], shuffle=False)
        else:
            # Data loaders for other models
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=config['batch_size'], shuffle=False)

        # Initialize the model, optimizer, scheduler, and trainer
        if config['model'] == 'resnet':
            pretrained = config.get('pretrained', True)
            if config['pretrained_model'] == 'resnet18':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet18(weights=ResNet18_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet18(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet34':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet34(weights=ResNet34_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet34(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet50':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet50(weights=ResNet50_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet50(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet101':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet101(weights=ResNet101_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet101(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet152':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet152(weights=ResNet152_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet152(weights=None), dropout_rate=config['dropout_rate']).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
            scheduler = utils.configure_scheduler(optimizer, config)

            trainer = ResNetTrainer(criterion=torch.nn.MSELoss(),
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    config=config,
                                    log_path=f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/logs/train_log.txt",
                                    model=model)

        elif config['model'] == 'vit':
            pretrained = config.get('pretrained', True)
            if config['pretrained_model'] == 'google/vit-base-patch16-224':
                if pretrained:
                    vit_base = ViTModel.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch16-224')
                else:
                    vit_config = ViTModel.config_class.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch16-224')
                    vit_base = ViTModel(vit_config)
                # model = vit.CustomViTModel(
                #     vit_base, dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'google/vit-base-patch32-224-in21k':
                if pretrained:
                    vit_base = ViTModel.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch32-224-in21k')
                else:
                    vit_config = ViTModel.config_class.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch32-224-in21k')
                    vit_base = ViTModel(vit_config)
                # model = vit.CustomViTModel(
                #     vit_base, dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'google/vit-large-patch16-224':
                if pretrained:
                    vit_base = ViTModel.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-large-patch16-224')
                else:
                    vit_config = ViTModel.config_class.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/hub/models--google--vit-large-patch16-224')
                    vit_base = ViTModel(vit_config)
            else:
                raise ValueError(f"Unsupported pretrained model: {config['pretrained_model']}")
            
            model = vit.CustomViTModel(
                vit_base, dropout_rate=config['dropout_rate']).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
            scheduler = utils.configure_scheduler(optimizer, config)

            trainer = ViTTrainer(criterion=torch.nn.MSELoss(),
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 device=device,
                                 model=model,
                                 config=config,
                                 log_path=f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/logs/train_log.txt")

        elif config['model'] == 'swin': 
            pretrained = config.get('pretrained', True)
            if config['pretrained_model'] == 'swin_small':
                if pretrained:
                    swin_model = SwinModel.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/microsoft/swin-small-patch4-window7-224')
                else:
                    swin_config = SwinModel.config_class.from_pretrained(
                        r'/home/wangzh/.cache/huggingface/microsoft/swin-small-patch4-window7-224')
                    swin_model = SwinModel(swin_config)
            elif config['pretrained_model'] == 'swin_base':
                if pretrained:
                    swin_model = SwinModel.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-base-patch4-window7-224')
                else:
                    swin_config = SwinModel.config_class.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-base-patch4-window7-224')
                    swin_model = SwinModel(swin_config)
            else:
                raise ValueError(f"Unsupported pretrained model: {config['pretrained_model']}")

            model = CustomSwinTransformer(swin_model=swin_model, dropout_rate=config['dropout_rate']).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
            scheduler = utils.configure_scheduler(optimizer, config)

            trainer = SwinTrainer(criterion=torch.nn.MSELoss(),
                                optimizer=optimizer,
                                scheduler=scheduler,
                                device=device,
                                config=config,
                                log_path=f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/logs/train_log.txt",
                                model=model)

        elif config['model'] == 'gin':
            model = gin.GIN(
                input_dim=3,
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout_rate=dropout_rate
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
            scheduler = utils.configure_scheduler(optimizer, config)

            trainer = GinTrainer(
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                model=model,
                config=config,
                log_path=f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/logs/train_log.txt"
            )

        elif config['model'] == 'moe':
            # Create experts
            experts = create_expert_models(device)
            # Initialize MoE model
            input_dim = 224 * 224 * 3
            model = moe.MoEModel(experts=experts, input_dim=input_dim).to(device)
            # Initialize optimizers and schedulers
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
            scheduler = utils.configure_scheduler(optimizer, config)

            trainer = MoETrainer(
                model=model,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                log_path=f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/logs/train_log.txt"
            )

        else:
            raise ValueError("Unsupported model type")

        # Set the model save path
        model_save_path = os.path.join(
            checkpoint_dir_base, f'model_fold_{fold + 1}.pth')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        print(f'Starting training for Fold {fold + 1}')
        metrics_history = trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['epochs'],
            model_save_path=model_save_path
        )
        print(f'Finished training for Fold {fold + 1}')

        # Setting the results save directory
        if config['model'] in ['resnet', 'vit', 'swin']: 
            results_dir = f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/results/{config['pretrained_model']}"
            results_dir = os.path.join(results_dir, f'fold_{fold + 1}')
            os.makedirs(results_dir, exist_ok=True)
        else:
            results_dir = f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/results"
            results_dir = os.path.join(results_dir, f'fold_{fold + 1}')
            os.makedirs(results_dir, exist_ok=True)

        # extract history from metrics_history
        train_mae_history = metrics_history['train_mae_history']
        val_mae_history = metrics_history['val_mae_history']
        train_mse_history = metrics_history['train_mse_history']
        val_mse_history = metrics_history['val_mse_history']
        train_rmse_history = metrics_history['train_rmse_history']
        val_rmse_history = metrics_history['val_rmse_history']

        save_path = os.path.join(results_dir, 'training_metrics.png')
        csv_path = os.path.join(results_dir, 'training_metrics.csv')

        # Call the plotting function
        utils.plot_training_metrics(
            train_mae_history, val_mae_history,
            train_mse_history, val_mse_history,
            train_rmse_history, val_rmse_history,
            save_path=save_path,
            csv_path=csv_path
        )

        # Load optimal model weights
        print(f"Loading best model weights for Fold {fold + 1}")
        model.load_state_dict(torch.load(model_save_path))

        print(f'Evaluating model on validation set for Fold {fold + 1}')
        if config['model'] == 'gin':
            val_predictions, val_actuals = test.evaluate_model_gin(
                model, val_loader, label_mean_list[fold], label_std_list[fold])
            train_predictions, train_actuals = test.evaluate_model_gin(
                model, train_loader, label_mean_list[fold], label_std_list[fold])
        else:
            val_predictions, val_actuals = test.evaluate_model(
                model, val_loader, label_mean_list[fold], label_std_list[fold])
            train_predictions, train_actuals = test.evaluate_model(
                model, train_loader, label_mean_list[fold], label_std_list[fold])

        fold_mape_plot_path = os.path.join(
            results_dir, f'mape_plot_fold_{fold + 1}.png')
        fold_mape_csv_path = os.path.join(
            results_dir, f'mape_data_fold_{fold + 1}.csv')
        utils.calculate_and_plot_mape(
            train_actuals, train_predictions, val_actuals, val_predictions,
            mape_limit=3,
            save_path=fold_mape_plot_path,
            csv_path=fold_mape_csv_path
        )

        all_train_actuals.extend(train_actuals)
        all_train_predictions.extend(train_predictions)
        all_test_actuals.extend(val_actuals)
        all_test_predictions.extend(val_predictions)

        csv_path = os.path.join(results_dir, f'metrics_{fold + 1}.csv')
        val_metrics = utils.calculate_and_print_metrics(
            train_actuals, train_predictions, val_actuals, val_predictions, csv_path)

        results[fold] = val_metrics
        print(f'Results for Fold {fold + 1}: {val_metrics}')

        torch.save(model.state_dict(), model_save_path)
        print(f'Model for Fold {fold + 1} saved at {model_save_path}')

        data = {
            'train_predictions': pd.Series(train_predictions),
            'train_actuals': pd.Series(train_actuals),
            'val_predictions': pd.Series(val_predictions),
            'val_actuals': pd.Series(val_actuals)
        }
        df = pd.DataFrame(data)
        output_train_test_path = os.path.join(results_dir, 'train_test.csv')
        df.to_csv(output_train_test_path, index=False)
        print(f"Train and validation predictions saved to {output_train_test_path}")

        test_r_squared = val_metrics['test_r_squared']
        test_mae = val_metrics['test_mae']
        test_mse = val_metrics['test_mse']
        test_rmse = val_metrics['test_rmse']
        output_Train_Test_path = os.path.join(
            results_dir, 'Train_and_Validation_Predictions_vs_Actuals.png')
        utils.plot_predictions_vs_actuals(
            train_actuals, train_predictions, val_actuals, val_predictions,
            test_r_squared, test_mae, test_mse, test_rmse,
            save_path=output_Train_Test_path
        )

    # Calculate the average result of all folds
    avg_metrics = utils.aggregate_cross_validation_results(results)
    print('Cross-validation results:')
    print(avg_metrics)

    mean_list_tensor = [torch.tensor(m) if isinstance(
        m, np.ndarray) else m for m in mean_list]
    std_list_tensor = [torch.tensor(s) if isinstance(
        s, np.ndarray) else s for s in std_list]

    final_mean = torch.mean(torch.stack(mean_list_tensor), dim=0)
    final_std = torch.mean(torch.stack(std_list_tensor), dim=0)

    final_label_mean = sum(label_mean_list) / len(label_mean_list)
    final_label_std = sum(label_std_list) / len(label_std_list)

    print(f"Final model mean: {final_mean}")
    print(f"Final model std: {final_std}")
    print(f"Final label mean: {final_label_mean}")
    print(f"Final label std: {final_label_std}")

    # Transformation of the preparatory reasoning stage
    if config['model'] in ['resnet', 'vit', 'moe', 'swin']: 
        _, inference_transform, _, _ = dataloader.prepare_dataset(
            config['train_annotations'],
            config['train_images'],
            config['batch_size'],
            device,
            train_indices=[], 
            is_training=False,
            mean=final_mean.tolist(), 
            std=final_std.tolist()    
        )
    elif config['model'] == 'gin':
        _, inference_transform, _, _, label_mean, label_std = dataloader.prepare_dataset_gin(
            config['train_annotations'],
            config['train_images'],
            config['batch_size'],
            device,
            train_indices=[], 
            is_training=False,
            mean=final_mean.tolist(),
            std=final_std.tolist(),
            label_mean=final_label_mean,
            label_std=final_label_std
        )

    models = []
    # Final assessments and projections
    for fold in range(k_folds):
        if config['model'] == 'resnet':
            pretrained = config.get('pretrained', True)
            if config['pretrained_model'] == 'resnet18':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet18(weights=ResNet18_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet18(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet34':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet34(weights=ResNet34_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet34(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet50':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet50(weights=ResNet50_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet50(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet101':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet101(weights=ResNet101_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet101(weights=None), dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'resnet152':
                if pretrained:
                    model = resnet.CustomResNetModel(
                        resnet152(weights=ResNet152_Weights.DEFAULT), dropout_rate=config['dropout_rate']).to(device)
                else:
                    model = resnet.CustomResNetModel(
                        resnet152(weights=None), dropout_rate=config['dropout_rate']).to(device)
            else:
                raise ValueError("Unsupported resnet model type")

        elif config['model'] == 'vit':
            # Initialize vit model
            if config['pretrained_model'] == 'google/vit-base-patch16-224':
                vit_base = ViTModel.from_pretrained(
                    r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch16-224')
                # model = vit.CustomViTModel(
                #     vit_base, dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'google/vit-base-patch32-224-in21k':
                vit_base = ViTModel.from_pretrained(
                    r'/home/wangzh/.cache/huggingface/hub/models--google--vit-base-patch32-224-in21k')
                # model = vit.CustomViTModel(
                #     vit_base, dropout_rate=config['dropout_rate']).to(device)
            elif config['pretrained_model'] == 'google/vit-large-patch16-224':
                vit_base = ViTModel.from_pretrained(
                    r'/home/wangzh/.cache/huggingface/hub/models--google--vit-large-patch16-224')
            model = vit.CustomViTModel(
                vit_base, dropout_rate=config['dropout_rate']).to(device)

        elif config['model'] == 'swin':
            pretrained = config.get('pretrained', True)
            if config['pretrained_model'] == 'swin_small':
                if pretrained:
                    swin_model = SwinModel.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-small-patch4-window7-224')
                else:
                    swin_config = SwinModel.config_class.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-small-patch4-window7-224')
                    swin_model = SwinModel(swin_config)
            elif config['pretrained_model'] == 'swin_base':
                if pretrained:
                    swin_model = SwinModel.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-base-patch4-window7-224')
                else:
                    swin_config = SwinModel.config_class.from_pretrained(r'/home/wangzh/.cache/huggingface/microsoft/swin-base-patch4-window7-224')
                    swin_model = SwinModel(swin_config)
            else:
                raise ValueError(f"Unsupported pretrained model: {config['pretrained_model']}")


        elif config['model'] == 'moe':
            # Initialize moe model
            experts = create_expert_models(device)
            input_dim = 224 * 224 *3
            model = moe.MoEModel(experts=experts, input_dim=input_dim).to(device)

        elif config['model'] == 'gin':
            model = gin.GIN(
                input_dim=3,
                hidden_dims=hidden_dims,
                output_dim=1,
                dropout_rate=dropout_rate
            ).to(device)

        model_save_path = os.path.join(
            checkpoint_dir_base, f'model_fold_{fold + 1}.pth')
        model.load_state_dict(torch.load(model_save_path), strict=False)
        model.eval()
        models.append(model)

    if config['model'] in ['resnet', 'vit', 'swin']: 
        results_dir = os.path.join(
                f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/results", config['pretrained_model'])
    else:
        # For other models, use the base directory
        results_dir = f"/home/wangzh/hydro_channel/hydro_channel/experiments/{config['model']}/results"

    output_predictions_path = os.path.join(results_dir, 'predictions.csv')

    if config['model'] == 'gin':
        # Predictions using ensemble
        predict.ensemble_predict_images_in_folder_gin(
            models=models,
            folder_path=config['test_images_folder'],
            label_mean=final_label_mean,
            label_std=final_label_std,
            transform=inference_transform,
            device=device,
            output_csv_path=output_predictions_path
        )
    else:
        predict.ensemble_predict_images_in_folder(
            models=models,
            folder_path=config['test_images_folder'],
            label_mean=final_label_mean,
            label_std=final_label_std,
            transform=inference_transform,
            device=device,
            output_csv_path=output_predictions_path
        )
        # output_predictions_path = os.path.join(results_dir, 'predictions_random.csv')
        # predict.ensemble_predict_images_in_folder(
        #     models=models,
        #     folder_path=config['random_images_folder'],
        #     label_mean=dataset.label_mean,
        #     label_std=dataset.label_std,
        #     transform=inference_transform,  # Use inference transform
        #     device=device,
        #     output_csv_path=output_predictions_path
        # )

    # Calculate assessment indicators, save charts, etc.
    metrics_csv_path = os.path.join(results_dir, 'metrics.csv')
    utils.calculate_and_print_metrics(
        all_train_actuals, all_train_predictions, all_test_actuals, all_test_predictions, metrics_csv_path)

    output_CP_plot_path = os.path.join(
        results_dir, 'Cumulative_probability_plots.png')
    output_CP_csv_path = os.path.join(
        results_dir, 'Cumulative_probability_data.csv')

    test_r_squared, test_mae, test_mse, test_rmse = utils.calculate_and_plot_mape(
        all_train_actuals, all_train_predictions, all_test_actuals, all_test_predictions,
        mape_limit=3,
        save_path=output_CP_plot_path,
        csv_path=output_CP_csv_path
    )

    output_Train_Test_path = os.path.join(
        results_dir, 'Train_and_Test_Predictions_vs_Actuals.png')
    utils.plot_predictions_vs_actuals(
        all_train_actuals, all_train_predictions, all_test_actuals, all_test_predictions,
        test_r_squared, test_mae, test_mse, test_rmse,
        save_path=output_Train_Test_path
    )

    data = {
        'train_predictions': pd.Series(all_train_predictions),
        'train_actuals': pd.Series(all_train_actuals),
        'test_predictions': pd.Series(all_test_predictions),
        'test_actuals': pd.Series(all_test_actuals)
    }
    df = pd.DataFrame(data)
    output_train_test_path = os.path.join(results_dir, 'train_test.csv')
    df.to_csv(output_train_test_path, index=False)
    print(f"Train and test predictions saved to {output_train_test_path}")


if __name__ == '__main__':
    main()
