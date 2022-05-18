from copy import copy
import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from src.utils import F1_Loss, Utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from src.dataset import TrainDataset, TestDataset
import time
import os
import copy
import pandas as pd
from torchsampler import ImbalancedDatasetSampler
import random

utils = Utils()
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

class Train():
    history = {
        'loss': [],
        'macro_f1': [],
        'val_loss': [],
        'val_macro_f1': [],
        'best train macro f1': (0, 0),
        'best val macro f1': (0, 0)
        }
    best_model_wts = {}

    def __init__(self, val_size=0.1, random_state=11) -> None:
        self.val_size = val_size
        self.random_state = random_state

    def train_one_epoch(self, loader, model, optimizer, loss_fn, scaler, device):
        model.train()
        losses = 0        
        y_pred_list , y_true_list= [], []
        loop = tqdm(loader)
        num_train_sample = len(loop)
        for batch_idx, (data, targets, _) in enumerate(loop):
          # save examples and make sure they look ok with the data augmentation,
          # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
          #save_image(data, f"hi_{batch_idx}.png")

            data = data.to(device=device)
            targets = targets.to(device=device)

            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                # forward
                y_pred= model(data)
                targets = torch.tensor(targets,dtype=torch.long)
                loss = loss_fn(y_pred, targets)

            # empty gradients
            optimizer.zero_grad()

            # gradient (backpropagation)
            scaler.scale(loss).backward()

            # update weights
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
            
            losses += loss.item()
            
            y_pred_list.extend(y_pred.cpu().data.numpy())
            y_true_list.extend(targets)
        
        y_pred_list =  torch.FloatTensor(y_pred_list)
        y_true_list = torch.as_tensor(y_true_list)
        train_macro_f1 = utils.macro_f1(y_pred_list, y_true_list, num_classes=self.num_classes)
        train_macro_f2 = utils.macro_f2(y_pred_list, y_true_list, num_classes=self.num_classes)
        # train loss per epoch
        epoch_loss_train = losses / num_train_sample
        return epoch_loss_train, train_macro_f1, train_macro_f2
    
    def check_accuracy(self, dataloader_val, model, loss_fn, device="cuda"):
        # evaluation mode
        model.eval()
        losses = 0
        y_pred_list , y_true_list= [], []
        filenames = []
        loop = tqdm(dataloader_val)
        num_eval_sample = len(loop)
        predictions = np.array([])
        with torch.no_grad():
            for x, y, filename in loop: 
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                losses += loss.item()
                # value, index
                v, pred = torch.max(y_pred, 1)

                pred = pred.cpu().numpy()

                predictions = np.concatenate((predictions, pred), axis = None)
                y_pred_list.extend(y_pred.cpu().data.numpy())
                y_true_list.extend(y)
                filenames.extend(filename)
        
        # calculate validation accuracy and loss for each epoch
        eval_macro_f1 = utils.macro_f1(torch.FloatTensor(y_pred_list), torch.as_tensor(y_true_list), num_classes=self.num_classes)
        eval_macro_f2 = utils.macro_f2(torch.FloatTensor(y_pred_list), torch.as_tensor(y_true_list), num_classes=self.num_classes)
        
        epoch_loss_eval = losses / num_eval_sample
        
        return epoch_loss_eval, eval_macro_f1, filenames, predictions, y_true_list, eval_macro_f2
    
    def getLabels(self, train_ds: Subset):
        return train_ds.dataset.data[train_ds.dataset.data.index.isin(list(train_ds.indices))].category_labels

    def getDataLoader(self, config, path_to_train_csv="", path_to_test_csv="", show=False,):
        if path_to_train_csv!="":
            train_ds = TrainDataset(
                path_to_csv=path_to_train_csv,
                transform=config.train_transforms,
            )
            val_ds = TrainDataset(
                path_to_csv=path_to_train_csv,
                transform=config.train_transforms,
            )

            indexes = list(range(len(train_ds)))

            train_indexes, val_indexes = train_test_split(indexes, test_size=self.val_size, random_state=self.random_state)

            data = {}
            data['train'] = Subset(train_ds, train_indexes)
            data['val'] = Subset(val_ds, val_indexes)
            train_loader = DataLoader(
                dataset = data['train'],
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                shuffle=False,
                sampler=ImbalancedDatasetSampler( data['train'], callback_get_label=self.getLabels )
            )
            val_loader = DataLoader(
                dataset = data['val'],
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                shuffle=False,
            )
        if show:
            self.__showBatch(config, train_loader)
            self.__showBatch(config, val_loader)
        #     dataiter = iter(train_loader)
        #     images, targets, _ = dataiter.next()
        #     for i in range(3):
        #         self.images_show(images, targets, mean=config.train_mean, std=config.train_std)
        #         images, targets, _ = dataiter.next()

        if path_to_test_csv!="":
            self.test_ds = TestDataset(
                path_to_csv=path_to_test_csv,
                transform=config.train_transforms,
            )
            self.test_loader = DataLoader(
                self.test_ds, 
                batch_size=config.BATCH_SIZE, 
                num_workers=config.NUM_WORKERS, 
                pin_memory=config.PIN_MEMORY,
                shuffle=False
            )
        
        return train_loader, val_loader


    def main(self, config, model, labels: dict, path_to_train_csv, flag=False):
        self.num_classes = len(labels)
        train_loader, val_loader = self.getDataLoader(config, path_to_train_csv=path_to_train_csv)

        # loss_fn = F1_Loss(num_classes=self.num_classes).cuda()
        loss_fn = nn.CrossEntropyLoss()
        self.loss_fn=loss_fn
        model = model.to(config.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler()

        if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
            if flag:
              print("Loading validation model")
              utils.load_checkpoint(torch.load("val.pth.tar"), model, optimizer, config.LEARNING_RATE)
              self.model = model
            else:
              utils.load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
              self.model = model
            #   Show weight
            # for param in model.parameters():
            #     print(param.data)

        for epoch in range(config.NUM_EPOCHS):
            epoch_start_time = time.time()
            epoch_loss_train, train_macro_f1, train_macro_f2 = self.train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)            

            eval_loss, eval_macro_f1, _, _, _, eval_macro_f2 = self.check_accuracy(val_loader, model, loss_fn, config.DEVICE)
            
            print(f'train loss: {epoch_loss_train:.4f}, train macro_f1: {train_macro_f1:.4f}, train_macro_f2: {train_macro_f2:.4f}')
            
            print(f' eval loss: {eval_loss:.4f}, eval macro_f1: {eval_macro_f1:.4f}, eval_macro_f2: {eval_macro_f2:.4f}')
            
            self.history['loss'].append(epoch_loss_train)
            self.history['macro_f1'].append(train_macro_f1)
                       
            self.history['val_loss'].append(eval_loss)
            self.history['val_macro_f1'].append(eval_macro_f1)
            
            
            print(f'epoch [{epoch+1:02d}/{config.NUM_EPOCHS}]: {time.time()-epoch_start_time:.2f} sec(s)')
            
            if config.SAVE_MODEL:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                utils.save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)
                
                 # find best macro f1 on training data
                if train_macro_f1 > self.history['best train macro f1'][1]:
                    print('*New best macro f1 on training data*')
                    # copy current model weights
                    self.best_model_wts = copy.deepcopy(model.state_dict())
                    self.history['best train macro f1'] = (epoch, train_macro_f1)

                 # find best macro f1 on validation data
                if (eval_macro_f1 > self.history['best val macro f1'][1]):
                    print('*New best macro f1 on validation data*')
                    self.history['best val macro f1'] = (epoch, eval_macro_f1)
                    utils.save_checkpoint(checkpoint, "val.pth.tar")
            self.model = model
    
    def plot_history(self):
        loss = self.history['loss']
        accuracy = self.history['macro_f1']
        val_loss = self.history['val_loss']
        val_accuracy = self.history['val_macro_f1']
        
        x = range(len(loss))

        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, accuracy, label='Training macro f1', color='#03045e', linewidth=2)
        if len(val_loss) != 0:
            plt.plot(x, val_accuracy, label='Validation macro f1', color='#48cae4', linewidth=2)
        plt.plot(self.history['best train macro f1'][0], 
                self.history['best train macro f1'][1], 
                'bo', label='Best train macro f1', markersize=7, color='black')
        plt.plot(self.history['best val macro f1'][0], 
                self.history['best val macro f1'][1], 
                'bo', label='Best val macro f1', markersize=7, color='black')
        plt.title('macro f1')
        plt.grid(True)
        plt.legend()
        
        
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, label='Training loss', color='#03045e', linewidth=2)
        if len(val_loss) != 0:
            plt.plot(x, val_loss, label='Validation loss', color='#48cae4', linewidth=2)
        plt.title('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    
    def testing(self, config, path_to_test_csv, submission=True):
        self.getDataLoader(config, path_to_test_csv=path_to_test_csv)
        test_loss, test_macro_f1, filenames, y_pred, y_true, test_macro_f2 = self.check_accuracy(self.test_loader, self.model, self.loss_fn, config.DEVICE)
        if not submission:
            print(f' test loss: {test_loss:.4f}, test macro_f1: {test_macro_f1:.4f}, test_macro_f2: {test_macro_f2:.4f}')
        else: 
            names = list(map(self.getName, filenames))
            df = pd.DataFrame({"img_name": names, "label": y_pred})
            df.to_csv("submission.csv", index=False)

    def getName(self, name):
        return name.split(".")[0]

    def __showBatch(self, config, dataloader):
        batches, labels, _ = next(iter(dataloader))
        print(range(0, len(dataloader)))
        sample = 12
        if sample>len(dataloader):
            sample = len(dataloader)
        i = int(sample**0.5)
        sample = i**2
        idx = random.sample(range(0, len(dataloader)), sample)
        batches, labels = batches[[idx]], labels[[idx]]
        fig, ax = plt.subplots(i, i, figsize=(15, 15))
        for index, (batch, label) in enumerate(zip(batches, labels)):
            batch, label = batch.to(config.DEVICE), label.to(config.DEVICE)
            print([index//i], [index%i])
            ax[index//i][index%i].imshow(batch.cpu().permute(1, 2, 0))
            pred = self.model(batch.unsqueeze(0)).argmax(-1).item()
            ax[index//i][index%i].title.set_text('Dog' if pred == 0 else 'Other')
            ax[index//i][index%i].axis('off')
        plt.show()