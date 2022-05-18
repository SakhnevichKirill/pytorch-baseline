import os
import sys
import src
from src.train import Train
from src.resnet18 import ResNet18
from src.config import Config
from src.preprocessing import Preprocessing
from src.Parsing.specific import Specific
import torch
from torch import nn
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


config = Config("b3.pth.tar")
# set_seed(config.SEED)

labels = {
    "dog": 0, 
    "other": 1,
}

path_to_train_csv = "./train.csv"
path_to_test_csv = "./test.csv"
# preprocessing = Preprocessing("./data", path_to_csv, labels, parsingType=1)
Preprocessing("./Dogs vs. Other", path_to_train_csv, labels, parsingType=1)
Preprocessing("./src/Parsing/ParsedImages", path_to_test_csv, labels, parsingType=1)


train = Train(val_size=0.1, random_state=11)
train_loader, val_loader = train.getDataLoader(config, path_to_train_csv=path_to_train_csv, path_to_test_csv=path_to_test_csv)
mean, std = config.get_mean_and_std(train_loader)
config.set_train_transforms(mean, std)
mean, std = config.get_mean_and_std(val_loader)
config.set_val_transforms(mean, std)
# print(config.train_transforms)
model = ResNet18()
dp_model = nn.DataParallel(model)


try:
    # TODO: добавить возможность в качестве параметра выбирать способ подгрузки весов
    train.main(config, dp_model, labels, path_to_train_csv, flag=True)
except KeyboardInterrupt:
    print('Interrupt')
    Specific(config, train)
    train.plot_history()
    
    # class labels needs be correct 
    # train.testing(config, path_to_test_csv, submission=False)
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
