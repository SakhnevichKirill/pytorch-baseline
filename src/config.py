import torch
# from torchvision import transforms 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

class Config():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 2e-3
    WEIGHT_DECAY = 1e-8
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    NUM_WORKERS = 2
    PIN_MEMORY = True
    SAVE_MODEL = True
    LOAD_MODEL = True
    RESIZE_WIDRTH = 230
    RESIZE_HEIGHT = 230
    WIDRTH_IMAGE = 224
    HEIGHT_IMAGE = 224
    SEED = 285399

    def __init__(self, CHECKPOINT_FILE):
        super().__init__()
        self.CHECKPOINT_FILE = CHECKPOINT_FILE
        self.set_train_transforms()
        self.set_val_transforms()
    
    def set_train_transforms(self, mean=[0, 0, 0], std=[1, 1, 1]):
        self.train_mean=mean
        self.train_std=std
        self.train_transforms = A.Compose(
            [
                A.Resize(width=self.RESIZE_WIDRTH, height=self.RESIZE_HEIGHT),
                A.RandomCrop(height=self.HEIGHT_IMAGE, width=self.WIDRTH_IMAGE),
                A.HorizontalFlip(p=0.5),# FLips the image w.r.t horizontal axis
                A.Blur(blur_limit=[3,3], p=0.5),
                A.CoarseDropout(max_holes=4, max_height=3, max_width=3, p=0.3),
                A.IAAAffine(shear=30, rotate=0, p=1, mode="constant"),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    def set_val_transforms(self, mean=[0, 0, 0], std=[1, 1, 1]):
        self.val_mean=mean
        self.val_std=std
        self.val_transforms = A.Compose(
            [
                A.Resize(height=self.HEIGHT_IMAGE, width=self.WIDRTH_IMAGE),
                A.Blur(blur_limit=[3,3], p=0.5),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    def get_mean_and_std(self, dataloader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _, _ in tqdm(dataloader):
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
        
        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std
  
# config1 = Config("b3_1.pth.tar")