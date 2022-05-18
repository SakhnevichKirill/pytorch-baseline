import torch
from torch import nn
from torch.nn import functional as F
class Utils():
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def macro_f2(self, y_pred, y_true, num_classes=10, beta=2):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes).to(torch.float32)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + self.epsilon)
        return F2.mean(0)

    def macro_f1(self, y_pred, y_true, num_classes=10):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, num_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        print("tp: ", tp.mean(),"tn: ",  tn.mean(),"fp: ",  fp.mean(),"fn: ",  fn.mean())
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        print("precision: ", precision.mean(),"recall: ",  recall.mean())
        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return f1.mean()
    
    # def base_metric(self, y_pred, y_true, num_classes=2):

    #     if prediction == 0:
    #         dog_lovers[group_id.item()]['dog'] += 1
    #     else:
    #         dog_lovers[group_id.item()]['other'] += 1

    def save_checkpoint(self, state, filename="my_checkpoint.pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)
        print("=> Ok")
    
    def load_checkpoint(self, checkpoint, model, optimizer, lr):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("=> Ok")

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, num_classes=10, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()