from typing import List, Tuple
from unicodedata import category
import pandas as pd
import os
import re
from src.config import Config
from src.dataset import TrainDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
import traceback
import logging


class Preprocessing():
    # parsingType = 1 - set class by path, example cat/1.jpg  => class="cat"
    # parsingType = 2 - set class by image name, example "cat.1.jpg", separator="." => class="cat"
    # parsingType = 3 - set class "" for submission
    def __init__(self, dirName:str, path_to_csv: str, labels: dict, parsingType=1, separator=".", fileType="jpg"):
        super().__init__()
        (dirs, categories, img_names) = self.getDataCategory(dirName, fileType=fileType, parsingType=parsingType, separator=separator)
        try:
            categories_labels = self.setCategoryLabels(categories, labels)
        except Exception as e:
            if parsingType != 3:
                unic_categories = set(categories)
                emp_labels = dict(zip(unic_categories, range(len(unic_categories))))
                categories_labels = self.setCategoryLabels(categories, emp_labels)
            else: 
                categories_labels=[0]*len(categories)
            logging.error(traceback.format_exc())
        data = self.createdf(dirs, img_names, categories, categories_labels)
        data.to_csv(path_to_csv, index=True)
        print(data.info())
        print(data["category"].value_counts())

    def getDataCategory(self, dirName: str, fileType="jpg", parsingType=1, separator=".")-> Tuple[List[str], List[str], List[str]]:
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(dirName)
        dirs = list()
        categories = list()
        img_names = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                recDirs, recCategories, recNames = self.getDataCategory(fullPath, fileType=fileType, parsingType=parsingType, separator=separator)
                dirs += recDirs
                categories += recCategories
                img_names += recNames
            else:
                fileEx = re.split('\.', entry)[-1]
                if (fileEx == fileType):
                    if parsingType==1:
                        category = self.__getDataPathCategory(dirName)
                    elif parsingType==2:
                        category = self.__getDataNameCategory(entry, separator)
                    # elif parsingType==2:
                    #     category = self.__getDataNameCategory(entry, separator)
                    else: category=""
                    dirs.append(dirName)
                    categories.append(category)
                    img_names.append(entry)
        return dirs, categories, img_names

    def __getDataPathCategory(self, dirName: str)-> str:
        category = re.split('/', dirName)[-1]
        return category

    def __getDataNameCategory(self, entry: str, separator: str)-> str:
        category = re.split('\%s'%separator, entry)[0]
        return category

    def culcCategoryLabels(self, categories: List[str])->List[int]:
        labels = list()
        categories_labels = list()
        for category in categories:
            if category not in labels:
                labels.append(category)
        for category in categories:
            categories_labels.append(labels.index(category))
        print(labels)
        return categories_labels

    def setCategoryLabels(self,categories: List[str], labels: dict)->List[int]:
        categories_labels = list()
        for category in categories:
            categories_labels.append(labels[category])
        return categories_labels

    def createdf(self, dirs: List[str], img_names: List[str], categories: List[str], categories_labels: List[int]):
        d = {'img_name': img_names, 'category': categories, "category_labels": categories_labels, "dirs": dirs}
        df = pd.DataFrame(data=d)
        return df

def getMeanStd():
    config = Config("b3.pth.tar")

    dataset = TrainDataset(
        path_to_csv="train.csv",
        transform=config.val_transforms,
    )

    indexes = list(range(len(dataset)))

    train_indexes, val_indexes = train_test_split(indexes, test_size=0.2)

    data = {}
    data['train'] = Subset(dataset, train_indexes)
    data['val'] = Subset(dataset, val_indexes)

    dataloader_train = DataLoader(dataset = data['train'], batch_size = config.BATCH_SIZE, num_workers=2, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(dataset = data['val'], batch_size = config.BATCH_SIZE, num_workers=2, shuffle=True, pin_memory=True)

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for image, label, file in tqdm(dataloader_train):
        data = torch.Tensor.float(image)
        channels_sum += torch.mean(data)
        channels_sqrd_sum += torch.mean(data**2)
        num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    print(mean, std)

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for image, label, file in tqdm(dataloader_train):
        data = torch.Tensor.float(image)
        channels_sum += torch.mean(data)
        channels_sqrd_sum += torch.mean(data**2)
        num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    print(mean, std)