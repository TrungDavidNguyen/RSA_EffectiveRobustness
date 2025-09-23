import warnings
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from net2brain.architectures.netsetbase import NetSetBase
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

import pathlib


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = pathlib.Path(folder_path)
        self.file_list = sorted(list(self.folder_path.glob('*.png')) + list(self.folder_path.glob('*.jpg')))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, str(img_path.stem)


# FeatureExtractor class
class FeatureExtractor:
    def __init__(self,
                 model,
                 netset=None,
                 netset_fallback="Standard",
                 device="cpu",
                 pretrained=True,
                 preprocessor=None,
                 extraction_function=None,
                 feature_cleaner=None):
        # Parameters
        self.model_name = model
        self.device = device
        self.pretrained = pretrained

        # Get values for editable functions
        self.preprocessor = preprocessor
        self.extraction_function = extraction_function
        self.feature_cleaner = feature_cleaner

        self.use_mixed_precision = True  # Enable mixed precision by default

        if netset is not None:
            self.netset_name = netset
            self.netset = NetSetBase.initialize_netset(self.model_name, netset, device)

            # Initiate netset-based functions
            self.model = self.netset.get_model(self.pretrained)
            self.layers_to_extract = self.netset.layers

        else:
            if isinstance(model, str):
                raise ValueError("If no netset is given, the model_name parameter needs to be a ready model")
            else:
                # Initiate as the Netset structure of choice in case user does not select preprocessing, extractor, etc.
                self.netset = NetSetBase.initialize_netset(
                    model_name=None, netset_name=netset_fallback, device=self.device
                )
                self.model = model
                self.model.eval()
                self.netset.loaded_model = self.model

    def extract(self, data_path, layers_to_extract=None):
        """
        Now with batch processing and correct shape
        """
        all_features = defaultdict(dict)

        dataset = ImageFolderDataset(data_path)
        torch.manual_seed(42)
        file_order = [str(f.stem) for f in dataset.file_list]

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()
        # Make sure model is on GPU
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        batch_features = defaultdict(list)
        self.data_type = "image"

        with torch.no_grad():
            for imgs, names in loader:
                imgs = imgs.to(self.device)
                if self.preprocessor is not None:
                    imgs = self.preprocessor(imgs, self.model_name, self.device)
                with torch.amp.autocast(enabled=self.use_mixed_precision, device_type='cuda'):
                    if self.extraction_function is None:
                        features = self.netset.extraction_function(imgs, layers_to_extract)
                    else:
                        features = self.extraction_function(imgs, layers_to_extract, model=self.model)

                    if self.feature_cleaner is None:
                        feature_cleaner = self.netset.get_feature_cleaner(self.data_type)
                        features = feature_cleaner(self.netset, features)
                    else:
                        features = self.feature_cleaner(features)

                    # Store batch features
                    for layer, value in features.items():
                        # Process entire batch at once
                        feats = value.detach().cpu().float().numpy()
                        # Add batch dimension to all features at once
                        feats = feats[:, np.newaxis, ...]
                        # Create name-feature pairs for the batch
                        batch_features[layer].extend(zip(names, feats))

                del features
                torch.cuda.empty_cache()

        # Reorganize in original order
        for layer in batch_features:
            temp_dict = dict(batch_features[layer])
            for name in file_order:
                if name in temp_dict:
                    all_features[layer][name] = temp_dict[name]

        return all_features
