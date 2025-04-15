import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import random_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from torch.optim import lr_scheduler
import random
import math
from glob import glob
import utils
import cv2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

utils.set_seed(3407)

class ResNetModified(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
        self.dropout = torch.nn.Dropout(0)
        num_ftrs = pretrained_model.fc.in_features
        self.fc = torch.nn.Linear(num_ftrs, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)  # flatten the image
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean.cpu().numpy(), std.cpu().numpy()

# Use the DataLoader to load the data and calculate the mean and standard deviation of the training set.
transform_for_mean_std = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  
])

dataset_for_mean_std = utils.CustomImageDataset_Z(
    annotations_file=r'E:\desktop\hydro_channel\dataset\train_images_zhenlie_black.xlsx',
    img_dir=r'E:\desktop\hydro_channel\dataset\gradcam\train_images_zhenlie_black',
    transform=transform_for_mean_std
)

data_loader_for_mean_std = DataLoader(dataset_for_mean_std, batch_size=4, shuffle=False)

mean, std = calculate_mean_std(data_loader_for_mean_std)
print(f"Calculated mean: {mean}, std: {std}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist()) 
])

# Load the dataset
dataset = utils.CustomImageDataset_Z(
    annotations_file=r'E:\desktop\hydroproj\train_images_zhenlie_black.xlsx',
    img_dir=r'E:\desktop\hydroproj\train_images_zhenlie_black',
    transform=transform
)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total_size = len(dataset)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


model = ResNetModified(resnet18(weights=ResNet18_Weights.DEFAULT))
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs=50, model_save_path="best_model.pth"):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    train_mae_history = []
    train_mse_history = []
    train_rmse_history = []

    val_mae_history = []
    val_mse_history = []
    val_rmse_history = []

    early_stopping = utils.EarlyStopping(patience=15, verbose=True, delta=0.00001)  

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # Include .logits for consistency with the reference code
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            all_train_labels.extend(labels.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        train_mae = mean_absolute_error(all_train_labels, all_train_outputs)
        train_mse = mean_squared_error(all_train_labels, all_train_outputs)
        train_rmse = math.sqrt(train_mse)

        train_mae_history.append(train_mae)
        train_mse_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        
        model.eval()
        running_loss = 0.0
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Include .logits for consistency with the reference code
                loss = criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item()
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().detach().numpy())

        val_loss = running_loss / len(val_loader)
        val_losses.append(val_loss)
        
        val_mae = mean_absolute_error(all_val_labels, all_val_outputs)
        val_mse = mean_squared_error(all_val_labels, all_val_outputs)
        val_rmse = math.sqrt(val_mse)

        val_mae_history.append(val_mae)
        val_mse_history.append(val_mse)
        val_rmse_history.append(val_rmse)

        scheduler.step()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
        
        early_stopping(val_loss, model, model_save_path)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    save_path1 = "result/resnet_pretrain/mse3/Training_and_Validation_Loss.png"
    utils.plot_training_metrics(train_mae_history, val_mae_history, train_mse_history, val_mse_history, train_rmse_history, val_rmse_history, save_path=save_path1)

def evaluate_model(model, test_loader, label_mean, label_std):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = outputs.cpu().data.numpy()

            predicted_original = predicted * label_std + label_mean
            actual_original = labels.cpu().numpy() * label_std + label_mean
            
            predictions.extend(predicted_original.flatten().tolist())
            actuals.extend(actual_original.flatten().tolist())
    
    return predictions, actuals


model_save_path="result/resnet_pretrain/mse3/best_model.pth"
# Train the model and save the best model
train_model(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=100, model_save_path=model_save_path)

# Load the best saved model
model.load_state_dict(torch.load(model_save_path))

# # Select target convolutional layers for Grad-CAM analysis
# target_layer = model.features[5]

# # Generate Grad-CAM visualization of multiple random images
#visualize_multiple_images_with_gradcam(model, dataset, target_layer, device, num_images=10)

train_predictions, train_actuals = evaluate_model(model, train_loader, dataset.label_mean, dataset.label_std)
test_predictions, test_actuals = evaluate_model(model, test_loader, dataset.label_mean, dataset.label_std)

utils.calculate_and_print_metrics(train_actuals, train_predictions, test_actuals, test_predictions)

save_path2 = "result/resnet_pretrain/mse3/Cumulative_probability_plots.png"
test_r_squared, test_mae, test_mse, test_rmse = utils.calculate_and_plot_mape(train_actuals, train_predictions, test_actuals, test_predictions, mape_limit=3, save_path=save_path2)
save_path3 = "result/resnet_pretrain/mse3/Train_and_Test_Predictions_vs_Actuals_with_Units.png"
utils.plot_predictions_vs_actuals(train_actuals, train_predictions, test_actuals, test_predictions, test_r_squared, test_mae, test_mse, test_rmse, save_path=save_path3)

# Define the function for predicting images in a folder using Z-score normalization
def predict_images_in_folder(model, folder_path, label_mean, label_std, output_csv_path=None):
    model.eval()
    predictions = {}

    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("Found image files:", image_paths)
    
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Apply the same transform used during training
        with torch.no_grad():
            output = model(image)
        normalized_value = output.cpu().numpy().flatten()[0]

        original_value = normalized_value * label_std + label_mean
        predictions[os.path.basename(image_path)] = original_value
    
    if output_csv_path:
        pd.DataFrame(list(predictions.items()), columns=['Image', 'Predicted Value']).to_csv(output_csv_path, index=False)
    
    return predictions

# Assuming the model has been trained and evaluated
folder_path = r'E:\desktop\hydroproj\test_images_zhenlie_black'
predictions = predict_images_in_folder(model, folder_path, label_mean=dataset.label_mean, label_std=dataset.label_std, output_csv_path=r'E:\desktop\hydroproj\result\resnet_pretrain\mse3\predictions_new_zhenlie.csv')

gradcam_output_dir = r'image_gradcam'
os.makedirs(gradcam_output_dir, exist_ok=True)

def save_gradcam_image(image_name, cam, img):
    # Save Grad-CAM superimposed images
    cam_on_image = utils.visualize_cam_on_image(img, cam)
    output_path = os.path.join(gradcam_output_dir, image_name)
    cv2.imwrite(output_path, cam_on_image)
    print(f"Saved Grad-CAM visualization to {output_path}")

def apply_gradcam_and_save_from_folder(model, folder_path, target_layer, device, save_folder):
    model.eval()

    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, image_path in enumerate(image_paths):

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  
        
        cam = utils.apply_gradcam(model, image_tensor.squeeze(0), target_layer, device)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) 
        
        image_name = os.path.basename(image_path)
        
        save_gradcam_image(image_name, cam, image_np)

# Select target convolutional layers for Grad-CAM analysis
target_layer = model.features[7]

apply_gradcam_and_save_from_folder(model, folder_path, target_layer, device, gradcam_output_dir)

# Print all image prediction results
for image_name, predicted_value in predictions.items():
    print(f"{image_name}: {predicted_value}")
