from torchvision import datasets, transforms
import torch
import json
from torch.utils.data import Subset
from tqdm import tqdm


def measure_accuracy(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Predictions for batch
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()  # Add num of correct predictions
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy


def measure_accuracy_o(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    def crop_border(img):
        """
        Crop 2 pixels border from image (like in objectnet paper)
        """
        width, height = img.size
        crop_area = (2, 2, width - 2, height - 2)
        return img.crop(crop_area)
    imagenet_to_objectnet = {
        409: 1, 530: 1, 414: 2, 954: 4, 419: 5, 790: 8, 434: 9, 440: 13, 703: 16, 671: 17, 444: 17,
        446: 20, 455: 29, 930: 35, 462: 38, 463: 39, 499: 40, 473: 45, 470: 46, 487: 48, 423: 52,
        559: 52, 765: 52, 588: 57, 550: 64, 507: 67, 673: 68, 846: 75, 533: 78, 539: 81, 630: 86,
        740: 88, 968: 89, 729: 92, 549: 98, 545: 102, 567: 109, 578: 83, 589: 112, 587: 115,
        560: 120, 518: 120, 606: 124, 608: 128, 508: 131, 618: 132, 619: 133, 620: 134, 951: 138,
        623: 139, 626: 142, 629: 143, 644: 149, 647: 150, 651: 151, 659: 153, 664: 154, 504: 157,
        677: 159, 679: 164, 950: 171, 695: 173, 696: 175, 700: 179, 418: 182, 749: 182, 563: 182,
        720: 188, 721: 190, 725: 191, 728: 193, 923: 196, 731: 199, 737: 200, 811: 201, 742: 205,
        761: 210, 769: 216, 770: 217, 772: 218, 773: 219, 774: 220, 783: 223, 792: 229, 601: 231,
        655: 231, 689: 231, 797: 232, 804: 235, 806: 236, 809: 237, 813: 238, 632: 239, 732: 248,
        759: 248, 828: 250, 850: 251, 834: 253, 837: 255, 841: 256, 842: 257, 610: 258, 851: 259,
        849: 268, 752: 269, 457: 273, 906: 273, 859: 275, 999: 276, 412: 284, 868: 286, 879: 289,
        882: 292, 883: 293, 893: 297, 531: 298, 898: 299, 543: 302, 778: 303, 479: 304, 694: 304,
        902: 306, 907: 307, 658: 309, 909: 310
    }
    crop_transform = transforms.Lambda(lambda img: crop_border(img))
    transform_with_crop = transforms.Compose([
        crop_transform,
        transform
    ])
    objectnet_mask = torch.tensor(list(imagenet_to_objectnet.keys()), device=device)
    imagenet_to_objectnet_tensor = torch.full((1000,), -1, dtype=torch.long, device=device)
    for k, v in imagenet_to_objectnet.items():
        imagenet_to_objectnet_tensor[k] = v

    # Load dataset containing classes without mapping to imagenet
    full_dataset = datasets.ImageFolder(path, transform=transform_with_crop)
    # objectnet indices that have mapping to imagenet
    valid_objectnet_labels = set(imagenet_to_objectnet.values())
    # indices of images that have mappings to imagenet
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in valid_objectnet_labels]
    dataset = Subset(full_dataset, filtered_indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            # Filter outputs to only contain imagenet classes that map to objectnet
            outputs = model(inputs)[:, objectnet_mask]
            predicted_indices = torch.argmax(outputs, dim=1)
            # map prediction indices of subset to their indices in full dataset
            imagenet_preds = objectnet_mask[predicted_indices]
            # map imagenet indices to objectnet
            objectnet_preds = imagenet_to_objectnet_tensor[imagenet_preds]

            correct += (objectnet_preds == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


def measure_accuracy_subset(model, path, transform, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(f'wnids/imagenet_{dataset}_wnids.json', 'r') as f:
        subset_wnids = json.load(f)
    # Mapping from wnid to index in the 1000-class output
    wnid_to_index = {wnid: idx for idx, wnid in enumerate(all_wnids)}
    # Convert subset wnids to corresponding indices in full ImageNet1K
    subset_indices = [wnid_to_index[wnid] for wnid in subset_wnids]

    subset_to_imagenet_tensor = torch.full((1000,), -1, dtype=torch.long, device=device)
    for idx, wnid in enumerate(subset_wnids):
        subset_to_imagenet_tensor[idx] = wnid_to_index[wnid]

    subset_mask = [wnid in subset_wnids for wnid in all_wnids]

    # Load dataset
    full_dataset = datasets.ImageFolder(path, transform=transform)
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in subset_indices]
    dataset = Subset(full_dataset, filtered_indices)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, subset_mask]  # Predictions for batch

            predicted = torch.argmax(outputs, dim=1)
            subset_preds = subset_to_imagenet_tensor[predicted]
            correct += (subset_preds == labels).sum().item()
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy


def measure_accuracy_a(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(r'wnids/imagenet_a_wnids.json', 'r') as f:
        imagenet_a_wnids = json.load(f)
    imagenet_a_mask = [wnid in imagenet_a_wnids for wnid in all_wnids]
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, imagenet_a_mask]  # Predictions for batch
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()  # Add num of correct predictions
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy


def measure_accuracy_r(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    with open(r'wnids/wnids.json', 'r') as f:
        all_wnids = json.load(f)
    with open(r'wnids/imagenet_r_wnids.json', 'r') as f:
        imagenet_r_wnids = json.load(f)
    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[:, imagenet_r_mask]  # Predictions for batch
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()  # Add num of correct predictions
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy