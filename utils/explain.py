import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import torch
from PIL import Image
from glob import glob

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self.model.eval()
        
        # Hook to save the gradients from the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, x):
        return self.model(x)

    def backward(self, target_output):
        self.model.zero_grad()
        target_output.backward(retain_graph=True)

    def generate_cam(self):
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (gradients * self.activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def apply_gradcam(model, image_tensor, target_layer, device):
    # Create GradCAM object
    grad_cam = GradCAM(model, target_layer)
    
    # Forward pass
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = grad_cam.forward(image_tensor)
    
    # Assume single regression value, get its gradient
    target_value = output
    grad_cam.backward(target_value)

    # Generate Grad-CAM heatmap
    cam = grad_cam.generate_cam()
    
    return cam


def visualize_cam_on_image(img, mask):
    # Resize the heatmap so that it is the same size as the input image
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0])) 

    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap onto original image
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)  
    # Displaying images and color bars
    fig, ax = plt.subplots()
    cam_image = np.uint8(255 * cam)
    im = ax.imshow(cam_image)
    # Create a colorbar using the original mask (not color mapped), mapped to the 0-1 range.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return cam_image

def visualize_multiple_images_with_gradcam(model, dataset, target_layer, device, num_images=5):
    model.eval()
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Randomly select a picture
        index = np.random.randint(0, len(dataset))
        test_image, _ = dataset[index]  # Getting images and tags from a dataset
        
        # Overlaying images with Grad-CAM
        cam = apply_gradcam(model, test_image, target_layer, device)
        test_image_np = test_image.permute(1, 2, 0).cpu().numpy()
        test_image_np = (test_image_np - test_image_np.min()) / (test_image_np.max() - test_image_np.min()) 
        
        # Overlaying Grad-CAM results onto the original image
        cam_on_image = visualize_cam_on_image(test_image_np, cam)
        
        # Displaying the Grad-CAM Diagram
        axs[i].imshow(cam_on_image)
        axs[i].axis('off')
    
    plt.show()

def compute_attention_entropy(attention_weights):
    """
    The attention entropy value of each head is calculated, indicating the degree of concentration of the attention distribution.
    """
    # attention_weights shape: [batch_size, num_heads, num_queries, num_keys]
    # Calculate entropy for each head
    attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
    
    # Average the batch and query dimensions to get the entropy value for each head
    return attention_entropy.mean(dim=[0, 2]).cpu().numpy()  # shape: [num_heads]

def save_entropy_values(entropy_values, image_name, save_dir='head_weight'):
    """
    Save the attentional entropy value for each head as a .csv file.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Naming the save file using the filename of the input image
    image_basename = os.path.splitext(os.path.basename(image_name))[0]  
    save_path = os.path.join(save_dir, f'{image_basename}_attention_entropy.csv')

    df = pd.DataFrame({'Head': range(len(entropy_values)), 'Entropy': entropy_values})
    df.to_csv(save_path, index=False)

    print(f"Entropy values saved to {save_path}")

def visualize_attention_entropy(entropy_values, image_name, save_dir='head_weight'):
    """
    Visualize the attentional entropy value for each head and save the image to a specified folder.
    Name the saved image using the filename of the input image.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_heads = len(entropy_values)
    plt.figure(figsize=(10, 6))

    bars = plt.bar(range(num_heads), entropy_values)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=10)
    
    plt.xlabel('Head Index')
    plt.ylabel('Entropy')
    plt.title('Attention Entropy of Each Head')

    # Generate a save path, named using the input image name
    image_basename = os.path.splitext(os.path.basename(image_name))[0] 
    save_path = os.path.join(save_dir, f'{image_basename}_attention_entropy.png')
    
    plt.savefig(save_path)
    plt.close() 

    print(f"Entropy plot saved to {save_path}")

def analyze_attention_entropy(attention_weights, image_tensor, image_path, save_dir1):
    """
    Analyzing the entropy of attention weights and generating entropy visualizations of the corresponding images
    """
    # Check if attention_weights is a tuple
    if isinstance(attention_weights, tuple):
        attention_weights = attention_weights[-1]  # Extract the last layer of attention weights

    # Make sure attention_weights is a tensor type
    if isinstance(attention_weights, torch.Tensor):
        # Average the attention weights of all heads
        attention_map = attention_weights.mean(dim=1)[0].detach().cpu().numpy()  
    else:
        raise ValueError("attention_weights is not a tensor after extraction")

    # Calculate and visualize entropy values
    entropy = -np.sum(attention_map * np.log(attention_map + 1e-10), axis=-1)
    visualize_attention_entropy(entropy, image_path, save_dir1)

    # Save entropy images
    entropy_image_path = os.path.join(save_dir1, f"entropy_{os.path.basename(image_path)}")
    print(f"Saved entropy image to {entropy_image_path}")


def visualize_attention_map_with_overlay_and_save(attention_weights, flow_channel_image_path, save_dir, image_size=224, patch_size=16, head_index=0):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if attention_weights is a tuple
    if isinstance(attention_weights, tuple):
        # Get the attentional weight of the last layer, assuming we're only dealing with the last layer
        attention_weights = attention_weights[-1] 

    # Check if the extracted attention_weights are tensor
    if isinstance(attention_weights, torch.Tensor):
        # Average the attention weights of all heads
        attention_map = attention_weights.mean(dim=1)[0].detach().cpu().numpy() 
    else:
        raise ValueError("attention_weights is not a tensor after extraction")

    # Print the shape of the attention weights
    print(f"Attention map shape: {attention_map.shape}") 
    
    # Remove the class token and keep only the attention value of the image patch
    attention_map = attention_map[1:, 1:]  
    num_patches = (image_size // patch_size) ** 2  
    attention_map = attention_map.mean(axis=0).reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches))) 

    # Resize the attention map to the same size as the input image
    attention_map_resized = cv2.resize(attention_map, (image_size, image_size))
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())

    # Load raw runner image
    flow_channel_image = Image.open(flow_channel_image_path).convert("RGB")
    flow_channel_image_resized = flow_channel_image.resize((image_size, image_size))

    # Create superimposed images
    fig, ax = plt.subplots()
    ax.imshow(flow_channel_image_resized, alpha=0.6) 
    im = ax.imshow(attention_map_resized, cmap='viridis', alpha=0.8)  

    # Associate the colorbar to the overlaid attention map
    plt.colorbar(im)
    plt.title(f"Self-Attention Map with Overlay - Head {head_index}")

    image_name = os.path.basename(flow_channel_image_path)
    save_path = os.path.join(save_dir, f"att_{image_name}")
    plt.savefig(save_path)
    plt.close() 

    print(f"Saved attention map overlay to {save_path}")

def predict_images_in_folder_with_attention(model, folder_path, output_csv_path=None, transform=None, device=None, save_dir1='head_weight', save_dir2='image_attention'):
    model.eval()
    predictions = {}

    # Get all image paths
    image_paths = glob(os.path.join(folder_path, "*.*"))
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("Found image files:", image_paths)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor, output_attentions=True)
            logits = outputs[0]  
            attention_weights = outputs[-1] 

        # Visualize self-attention with original runner image superimposed, save image to folder
        visualize_attention_map_with_overlay_and_save(attention_weights, image_path, save_dir2, image_size=224, patch_size=16)

        # Use extracted attention_weights instead of model.attentions
        analyze_attention_entropy(attention_weights, image_tensor, image_path, save_dir1)

        normalized_value = logits.cpu().numpy().flatten()[0]
        predictions[os.path.basename(image_path)] = normalized_value

    # If the output path is specified, the results are saved to a CSV file
    if output_csv_path:
        pd.DataFrame(list(predictions.items()), columns=['Image', 'Predicted Value']).to_csv(output_csv_path, index=False)

    return predictions
