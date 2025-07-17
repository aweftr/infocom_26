'''
储存常见功能
'''
import numpy as np
import torch


def linear_decay(epoch, max_epoch, eps_start, eps_end):
    return max(eps_end, eps_start - (eps_start - eps_end) * (epoch / max_epoch))

def trimmed_mean(data, percentage=0.1):
    ''' return the trimmed mean of data by excluding first percentage and last percentage, total 2*percentage data '''
    if not data:
        raise ValueError("The data list is empty")

    if not 0 <= percentage < 0.5:
        raise ValueError("Percentage must be between 0 and 0.5")

    n = len(data)
    k = int(n * percentage)
    
    # Sort the data
    sorted_data = sorted(data)
    
    # Remove the lowest and highest k elements
    trimmed_data = sorted_data[k:n-k]
    
    # Calculate the mean of the remaining data
    if not trimmed_data:
        raise ValueError("Trimmed data is empty, increase the data size or reduce the percentage")

    mean_value = sum(trimmed_data) / len(trimmed_data)
    return mean_value

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
