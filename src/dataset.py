import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import warnings

class TrainDataset(Dataset):
  def __init__(self, path_to_csv: str, transform=None):
    super().__init__()
    self.transform = transform
    self.data = pd.read_csv(path_to_csv)
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    [i, image_name, category, label, dir] = self.data.iloc[index]
    # print(image_name, category, label, dir)
    image = np.array(Image.open(os.path.join(dir, image_name)))
    # image = np.stack((image,)*3, axis=-1)
    # print(image.shape)
    if self.transform:
        image = self.transform(image=image)["image"]
    # print(image, label, image_name)
    return image, label, image_name

class TestDataset(Dataset):
  def __init__(self, path_to_csv: str, transform=None):
    super().__init__()
    self.transform = transform
    self.data = pd.read_csv(path_to_csv)
    self.csvCategorys = self.data.drop_duplicates(subset=['category'])
    print(self.csvCategorys)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    
    [i, image_name, category, label, dir] = self.data.iloc[index]
    # print(image_name, category, label, dir)
    image = np.array(Image.open(os.path.join(dir, image_name)))
    # image= np.stack((image,)*3, axis=-1)
    if (len(image.shape)!=3):
      image= np.stack((image,)*3, axis=-1)
    # print(len(image.shape))
    if self.transform:
        image = self.transform(image=image)["image"]
    return image, label, image_name
  