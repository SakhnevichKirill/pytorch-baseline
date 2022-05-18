import imp
import json
import pandas as pd
from src.train import Train
from src.config import Config
import torch
from tqdm import tqdm

class Specific():

    def __init__(self, config: Config, train: Train, path_to_csv="./result.csv"):
        dog_lovers = self.getBaseMetric(config, train)
        with open('./src/Parsing/public_info.json') as d:
            public_info = json.load(d)
        to_del = set(public_info.keys()) - set(dog_lovers.keys())
        for i in to_del:
            del public_info[i]

        df = self.createCsv(public_info, dog_lovers)
        df = df.sort_values(by='base_metric', ascending=False)
        df.to_csv(path_to_csv, index=True)
        print(df.head())

    def createCsv(self, public_info: dict, dog_lovers: dict):
        publics_df = pd.DataFrame(public_info).transpose().reset_index()
        dog_lovers_df = pd.DataFrame(dog_lovers).transpose().reset_index()
        df = pd.merge(dog_lovers_df, publics_df)
        df['base_metric'] = df['dog'] / (df['dog'] + df['other']) * df['mean_comments_number'] / df['subscribers_number']
        return df        
    
    def getBaseMetric(self, config: Config, train: Train):
        
        dog_lovers = {str(v): {'dog': 0, 'other': 0} for v in train.test_ds.csvCategorys.category}
        # print(dog_lovers)
        train.model.eval()
        with torch.no_grad():
            loop = tqdm(train.test_loader) 
            for batch_idx, (x, y, _) in enumerate(loop):
                # TODO: targets to categories
                batch, y = x.to(config.DEVICE), y.to(config.DEVICE)
                
                y_pred = train.model(batch)
                v, pred = torch.max(y_pred, 1)
                dog_lovers = self.getScore(pred.cpu().numpy(), y.cpu().numpy(), dog_lovers, train.test_ds.csvCategorys)
        print(dog_lovers)
        return dog_lovers

    
    def getScore(self, y_preds, y, dog_lovers, df):
        for y, y_pred in zip(y, y_preds):
            group_id = str(df.loc[df.category_labels == y].category.item())
            if y_pred == 0:
                dog_lovers[group_id]['dog'] += 1
            else:
                dog_lovers[group_id]['other'] += 1
        return dog_lovers
