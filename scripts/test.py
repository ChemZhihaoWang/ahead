import torch
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_model_gin(model, test_loader, label_mean, label_std):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader: 
            batch = batch.to(device)
            # Forward pass
            output = model(batch).squeeze(dim=-1)
            # Get predicted and actual values
            predicted = output.view(-1).cpu().numpy()
            actual = batch.y.view(-1).cpu().numpy()
            # Reverse normalization if necessary
            predicted_original = utils.reverse_normalization(predicted, label_mean, label_std)
            actual_original = utils.reverse_normalization(actual, label_mean, label_std)
            # Store the predictions and actual values
            predictions.extend(predicted_original.flatten().tolist())
            actuals.extend(actual_original.flatten().tolist())

    return predictions, actuals
