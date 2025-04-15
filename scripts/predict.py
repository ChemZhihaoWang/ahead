import os
import pandas as pd
import torch
from PIL import Image
from glob import glob
from dataset.dataloader import image_to_graph
import numpy as np 

def ensemble_predict_images_in_folder(models, folder_path, label_mean, label_std, transform, device, output_csv_path=None):
    # Set to evaluation mode
    for model in models:
        model.eval()

    predictions = {}

    # Get paths to all images in a folder
    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  

        model_outputs = []
        with torch.no_grad():
            for model in models:
                output = model(image)
                model_outputs.append(output.cpu().numpy().flatten()[0])

        normalized_value = np.mean(model_outputs)

        # inverse normalization process
        original_value = normalized_value * label_std + label_mean
        predictions[os.path.basename(image_path)] = original_value
    
    if output_csv_path:
        pd.DataFrame(list(predictions.items()), columns=['Image', 'Predicted Value']).to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")


def ensemble_predict_images_in_folder_gin(models, folder_path, label_mean, label_std, transform, device, output_csv_path=None):
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()

    predictions = {}

    # Get all image paths in the folder
    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).to(device)  # Apply the transform and move to the correct device

        # Convert the image to graph data
        graph_data = image_to_graph(image, torch.tensor([0.0]))  # `label` can be a placeholder as it's not used
        graph_data = graph_data.to(device)

        model_outputs = []
        with torch.no_grad():
            for model in models:
                output = model(graph_data).squeeze(dim=-1)
                model_outputs.append(output.cpu().numpy().flatten()[0])

        # Compute the average of the model outputs
        normalized_value = np.mean(model_outputs)

        # Inverse normalization to get the original scale
        original_value = normalized_value * label_std + label_mean
        predictions[os.path.basename(image_path)] = original_value

    if output_csv_path:
        pd.DataFrame(list(predictions.items()), columns=['Image', 'Predicted Value']).to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")

    return predictions
