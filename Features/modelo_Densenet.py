
import os
import torch
import torchvision
import torchvision.transforms as T
import pydicom
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 1. DenseNetFeatureExtractor Class
class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        base_model = torchvision.models.densenet121(pretrained=True)
        self.features = base_model.features
        self.eval()

        self.target_layers = ['denseblock3', 'denseblock4', 'norm5']
        self.activations = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, module in self.features.named_children():
            if name in self.target_layers:
                module.register_forward_hook(get_activation(name))

    def forward(self, x):
        with torch.no_grad():
            _ = self.features(x)
            pooled_feats = []
            for layer_name in self.target_layers:
                fmap = self.activations[layer_name]
                pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
                pooled_feats.append(pooled)
            features = torch.cat(pooled_feats, dim=1)
        return features

# 2. DICOMDataset Class
class DICOMDataset(Dataset):
    def __init__(self, imgs):
        self.images = imgs
        self.transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Ensure PIL RGB image for torchvision transforms
        if isinstance(img, np.ndarray):
            arr = img
            # Normalize/convert dtype to uint8 if necessary
            if arr.dtype != np.uint8:
                # Assume input floats in [0,1] or HU-clipped/normalized; clamp to [0,1] then scale
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255.0).round().astype(np.uint8)
            # Handle grayscale or 3-channel arrays
            if arr.ndim == 2:
                pil = Image.fromarray(arr, mode='L').convert('RGB')
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pil = Image.fromarray(arr, mode='RGB')
            else:
                raise TypeError(f"Unexpected numpy image shape: {arr.shape}")
        elif isinstance(img, Image.Image):
            pil = img.convert('RGB')
        else:
            raise TypeError(f"Unexpected image type: {type(img)}")

        tensor = self.transform(pil)
        return tensor
    
    def add_image(self, img):
        self.images.append(img)
    
    

# 3. Main pipeline execution
if __name__ == "__main__":
    # Define transforms
 

    
   
    dicom_dataset = DICOMDataset()
    dataloader = DataLoader(dicom_dataset, batch_size=8, shuffle=False)

    # Initialize feature extractor
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = DenseNetFeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    all_features = []
    for i, batch_images in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}")
        batch_images = batch_images.to(DEVICE)
        features = feature_extractor(batch_images)
        all_features.append(features.cpu())


    final_features = torch.cat(all_features, dim=0)
    df_features = pd.DataFrame(final_features.numpy())
    output_csv_path = "dicom_densenet_features.csv"
    df_features.to_csv(output_csv_path, index=False)

    print(f"Features extracted and saved to {output_csv_path}")
    print(f"Shape of final features: {final_features.shape}")

    # End of script
    pass

