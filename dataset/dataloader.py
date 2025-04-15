import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader


class CustomImageDataset_Z(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # Calculate mean and standard deviation of labels
        self.label_mean = self.img_labels.iloc[:, 1].mean()
        self.label_std = self.img_labels.iloc[:, 1].std()

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        # Get label
        label = self.img_labels.iloc[idx, 1]
        # Normalize label using Z-score standardization
        normalized_label = (label - self.label_mean) / self.label_std
        # Process image
        if self.transform:
            image = self.transform(image)
        # Convert label to tensor
        normalized_label = torch.tensor(normalized_label, dtype=torch.float)
        
        return image, normalized_label
    
def calculate_mean_std(loader, device):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean.cpu().numpy(), std.cpu().numpy()

def prepare_dataset(annotations_file, img_dir, batch_size, device, train_indices):
    # Prepare transformations for calculating mean and standard deviation
    transform_for_mean_std = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create complete dataset (without data augmentation)
    full_dataset = CustomImageDataset_Z(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=transform_for_mean_std
    )

    # Create training subset, only for calculating mean and standard deviation
    train_subset_for_mean_std = torch.utils.data.Subset(full_dataset, train_indices)

    # Calculate mean and standard deviation
    data_loader_for_mean_std = DataLoader(train_subset_for_mean_std, batch_size=batch_size, shuffle=False)
    mean, std = calculate_mean_std(data_loader_for_mean_std, device)
    print(f"Calculated mean: {mean}, std: {std}")

    # Define transformations with data augmentation, using mean and standard deviation calculated on current fold training set
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # Use new transform to create complete dataset
    dataset = CustomImageDataset_Z(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=transform
    )

    return dataset, transform


# Convert image dataset to graph dataset
def image_to_graph(image, label):
    h, w = image.shape[1], image.shape[2]
    edge_index = []
    for i in range(h):
        for j in range(w):
            if i > 0:
                edge_index.append([i * w + j, (i - 1) * w + j])
            if i < h - 1:
                edge_index.append([i * w + j, (i + 1) * w + j])
            if j > 0:
                edge_index.append([i * w + j, i * w + (j - 1)])
            if j < w - 1:
                edge_index.append([i * w + j, i * w + (j + 1)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = image.view(-1, 3)
    y = label.unsqueeze(0)
    return Data(x=x, edge_index=edge_index, y=y)


def prepare_dataset_gin(annotations_file, img_dir, batch_size, device, train_indices):
    # Prepare transformations for calculating image mean and standard deviation
    transform_for_mean_std = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    # Create complete dataset (for subset indexing, without data augmentation)
    full_dataset = CustomImageDataset_Z(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=transform_for_mean_std
    )

    # Create training subset, only for calculating image mean and standard deviation
    train_subset_for_mean_std = torch.utils.data.Subset(full_dataset, train_indices)

    # Calculate training set image mean and standard deviation
    data_loader_for_mean_std = DataLoader(train_subset_for_mean_std, batch_size=batch_size, shuffle=False)
    mean, std = calculate_mean_std(data_loader_for_mean_std, device)
    print(f"Calculated image mean: {mean}, std: {std}")

    label_mean = train_subset_for_mean_std.label_mean
    label_std = train_subset_for_mean_std.label_std
    print(f"Calculated label mean: {label_mean}, std: {label_std}")

    # Define transformations with data augmentation, using mean and standard deviation calculated on current fold training set
    transform = transforms.Compose([
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # Use new transform to create complete dataset
    dataset = CustomImageDataset_Z(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=transform
    )
    # Update dataset label mean and standard deviation (ensure consistency)
    dataset.label_mean = label_mean
    dataset.label_std = label_std

    # Convert dataset to graph dataset
    def dataset_to_graph_dataset(dataset):
        graph_data_list = []
        for image, label in dataset:
            graph_data = image_to_graph(image, label)
            graph_data_list.append(graph_data)
        return graph_data_list

    graph_dataset = dataset_to_graph_dataset(dataset)

    return graph_dataset, transform, label_mean, label_std
