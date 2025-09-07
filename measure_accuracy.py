import json

import torch
from torch.utils.data import Subset
from torchvision import datasets
from tqdm import tqdm


def evaluate_model(model, dataloader, device, output_mask=None, label_map=None):
    model.to(device)
    model.eval()
    correct = total = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if output_mask is not None:
                outputs = outputs[:, output_mask]

            predicted = torch.argmax(outputs, dim=1)

            if label_map is not None:
                predicted = label_map[predicted]

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100


def measure_accuracy(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, data_loader, device)


def get_mask_map_indices(dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    sub_wnids_path = f'wnids/imagenet_{dataset}_wnids.json'
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(sub_wnids_path, 'r') as f:
        subset_wnids = json.load(f)
    # Mapping from wnid to index in the 1000-class output
    wnid_to_index = {wnid: idx for idx, wnid in enumerate(all_wnids)}
    # Convert subset wnids to corresponding indices in full ImageNet1K
    subset_indices = [wnid_to_index[wnid] for wnid in subset_wnids]

    subset_to_imagenet_tensor = torch.full((1000,), -1, dtype=torch.long, device=device)
    for idx, wnid in enumerate(subset_wnids):
        subset_to_imagenet_tensor[idx] = wnid_to_index[wnid]

    subset_mask = [wnid in subset_wnids for wnid in all_wnids]

    return subset_mask, subset_to_imagenet_tensor, subset_indices


def measure_accuracy_subset_r(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    subset_mask, subset_to_imagenet_tensor, subset_indices = get_mask_map_indices("r")

    # Load dataset
    full_dataset = datasets.ImageFolder(path, transform=transform)
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in subset_indices]

    dataset = Subset(full_dataset, filtered_indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    return evaluate_model(model, data_loader, device, subset_mask, subset_to_imagenet_tensor)


def measure_accuracy_subset_a(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    subset_mask, subset_to_imagenet_tensor, subset_indices = get_mask_map_indices("a")

    # Load dataset
    full_dataset = datasets.ImageFolder(path, transform=transform)
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in subset_indices]

    dataset = Subset(full_dataset, filtered_indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    return evaluate_model(model, data_loader, device, subset_mask, subset_to_imagenet_tensor)


def measure_accuracy_a(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(r'wnids/imagenet_a_wnids.json', 'r') as f:
        imagenet_a_wnids = json.load(f)
    imagenet_a_mask = [wnid in imagenet_a_wnids for wnid in all_wnids]

    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, data_loader, device, imagenet_a_mask)


def measure_accuracy_r(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(r'wnids/imagenet_r_wnids.json', 'r') as f:
        imagenet_r_wnids = json.load(f)
    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]

    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_model(model, data_loader, device, imagenet_r_mask)