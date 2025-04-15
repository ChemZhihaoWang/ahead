import os
import cv2

from PIL import Image
import matplotlib.pyplot as plt
from utils.explain import GradCAM, apply_gradcam, visualize_cam_on_image

def save_gradcam_image(image_name, cam, img, gradcam_output_dir):
    """Saving Grad-CAM Overlay Images"""
    cam_on_image = visualize_cam_on_image(img, cam)
    output_path = os.path.join(gradcam_output_dir, image_name)
    cv2.imwrite(output_path, cam_on_image)
    print(f"Saved Grad-CAM visualization to {output_path}")

def apply_gradcam_and_save_from_folder(model, folder_path, target_layer, device, gradcam_output_dir, transform):
    """Apply Grad-CAM and save the image to the specified folder"""
    model.eval()
    
    image_paths = [p for p in os.listdir(folder_path) if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_path in image_paths:
        image = Image.open(os.path.join(folder_path, image_path)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        cam = apply_gradcam(model, image_tensor.squeeze(0), target_layer, device)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) 
        
        save_gradcam_image(image_path, cam, image_np, gradcam_output_dir)
