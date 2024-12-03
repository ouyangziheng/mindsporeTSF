import numpy as np
import mindtorch.torch as torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        """
        初始化早停机制。
        
        参数:
        - patience: 如果验证集损失连续 `patience` 轮没有改善，则停止训练
        - min_delta: 认为损失有改善的最小阈值，若小于此值认为没有改善
        - verbose: 是否打印每次早停检查的信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = np.inf  
        self.counter = 0 
        self.stop_training = False  
    
    def check(self, val_loss, model):
        """
        检查验证集损失是否有改善，并根据需要更新状态。
        
        参数:
        - val_loss: 当前验证集的损失
        - model: 当前的模型
        """
        if self.best_loss - val_loss > self.min_delta:  
            self.best_loss = val_loss 
            self.counter = 0  
            if self.verbose:
                print(f"验证集损失改善为 {val_loss:.4f}")
        else:  
            self.counter += 1  
            if self.verbose:
                print(f"验证集损失没有改善，当前损失为 {val_loss:.4f}")
        
        if self.counter >= self.patience:  
            self.stop_training = True  
            if self.verbose:
                print(f"验证集损失连续 {self.patience} 轮没有改善，停止训练")
    
    def reset(self):
        """重置早停机制，适用于每次新的训练开始时。"""
        self.best_loss = np.inf
        self.counter = 0
        self.stop_training = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
