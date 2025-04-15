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

output_csv_path = 'E:/desktop/hydro_channel/others/t_sne_results.csv'
tsne_df.to_csv(output_csv_path, index=False)
print(f"t-SNE results saved to: {output_csv_path}")

plt.figure(figsize=(18, 14)) 
scatter = plt.scatter(
    tsne_df['tSNE1'], 
    tsne_df['tSNE2'], 
    c=tsne_df['current'], 
    cmap='plasma',  
    s=240, 
    edgecolors='white',  
    linewidth=0.5,  
    alpha=0.8  
)

cbar = plt.colorbar(scatter)
cbar.set_label('Current Density (A cm⁻²)', fontsize=24, labelpad=15)  
cbar.ax.tick_params(labelsize=18, width=1.5, length=8)  
cbar.outline.set_linewidth(1.5)  

plt.title('t-SNE Feature Visualization by Current Density', fontsize=26, pad=20)
plt.xlabel('tSNE1', fontsize=22, labelpad=10)
plt.ylabel('tSNE2', fontsize=22, labelpad=10)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.grid(False)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.savefig('E:/desktop/hydro_channel/others/t_sne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
