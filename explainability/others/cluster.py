import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  

resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])  
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_vectors = []
currents = []  

data = pd.read_csv('E:/desktop/hydro_channel/dataset/hydrofile_t_SNE.csv')

# Iterate through each image and extract features while recording current values
for idx, row in data.iterrows():
    img_file = row['image']
    current_value = row['current']
    currents.append(current_value)
    
    img_path = os.path.join('E:/desktop/hydro_channel/dataset/hydrofig', img_file) 
    img = Image.open(img_path).convert('RGB')
    
    img_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        feature_vector = resnet(img_tensor).squeeze().numpy()
    
    image_vectors.append(feature_vector)

features_df = pd.DataFrame(image_vectors)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_df)

tsne_df = pd.DataFrame(tsne_results, columns=['tSNE1', 'tSNE2'])
tsne_df['current'] = currents

# Classification by current density: low (0-33%), medium (33%-66%), high (66%-100%)
current_percentiles = np.percentile(currents, [33, 66])
low_threshold, high_threshold = current_percentiles
tsne_df['category'] = pd.cut(
    tsne_df['current'],
    bins=[-np.inf, low_threshold, high_threshold, np.inf],
    labels=['Low Current Density', 'Medium Current Density', 'High Current Density']
)

tsne_df.to_csv('E:/desktop/hydro_channel/others/t_sne_current_categories.csv', index=False)

output_path = 'E:/desktop/hydro_channel/others/t_sne_current_categories.png'
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    tsne_df['tSNE1'],
    tsne_df['tSNE2'],
    c=tsne_df['category'].cat.codes,
    cmap='viridis',
    s=100,
    edgecolors='white',
    linewidth=0.5,
    alpha=0.8
)

handles, _ = scatter.legend_elements()
categories = tsne_df['category'].cat.categories.tolist()
plt.legend(handles=handles, labels=categories, fontsize=12, title='Current Density')

plt.title('t-SNE Feature Visualization with Current Density Categories', fontsize=14)
plt.xlabel('tSNE1', fontsize=12)
plt.ylabel('tSNE2', fontsize=12)
plt.grid(False)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()  

print(f"t-SNE visualization and categories saved to:")
print(f" - Image: {output_path}")
print(f" - CSV: E:/desktop/hydro_channel/others/t_sne_current_categories.csv")
