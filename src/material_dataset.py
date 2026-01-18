import torch
from PIL import Image

class MaterialDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform, classes, class_to_idx, labels):
        self.files = files
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label