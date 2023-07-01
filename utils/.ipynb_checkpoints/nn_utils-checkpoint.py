from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import torch; torch.manual_seed(42)
from torch import nn
import math

from torch.utils.data import Dataset, DataLoader

from .early_stopping import EarlyStopping

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class nn_dataset(Dataset):
    """
    To do some changes in dataset
    """
    def __init__(self,data):
        # print(data[:,:-48])
        x = data[:,:-48].astype('float32')
        # x = np.apply_along_axis(self.in_array_flattten, axis = 1, arr = data)
        
        y = data[:,-48:].astype('float32')
        
        self.x = torch.tensor(x,dtype=torch.float32).to(device)
        self.y = torch.tensor(y,dtype=torch.float32).to(device)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length
    
    def in_array_flattten(self, a):
        """Flatten array inside array"""    

        if self.train_fd:
            return a[0]
        return np.append(a[0].flatten(), a[1])


class nn_net(nn.Module):
    """
    NN model
    """
    def __init__(self,input_size, nn_arch: list):
        super(nn_net,self).__init__()
        
        self.layers = nn_arch
        

    def forward(self, x):
        x = self.layers(x)
        return x
    
class nn_fd(nn.Module):
    """
    Weidmann model
    """
    def __init__(self):
        super(nn_fd,self).__init__()
        
        self.p_v0 = nn.parameter.Parameter(torch.tensor(1.0))
        self.p_t = nn.parameter.Parameter(torch.tensor(1.0))
        self.p_l = nn.parameter.Parameter(torch.tensor(1.0))
        

    def forward(self, x):
        x = self.p_v0*(1-torch.exp((self.p_l-x)/(self.p_v0*self.p_t)))
        return x
    
    
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    
    Args:
        m (_type_): _description_
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            

def validation(model, valid_loader, loss_function):
    """_summary_

    Args:
        model (_type_): _description_
        valid_loader (_type_): _description_
        loss_function (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Settings
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for data in valid_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss_total += loss.item()

    return loss_total / len(valid_loader)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def AIC_score(model, n, loss):
    
    k = get_n_params(model)
    score = 2*k + n*np.log(loss) + n*(1 + np.log(math.pi))
    return score

def train(model, epochs, optimizer, trainloader, valloader, criterion):
    # Early stopping
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta = 0)
    for epoch in range(0, epochs):
        # Print epoch
        if True: #epoch % 10 == 0:
            print(f'Starting epoch {epoch+1}')
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)
            
            # print(outputs.shape, targets.shape, train_fd )
            # Compute loss
            # if train_fd:
            #     outputs = outputs.view(len(outputs),1)
            #     print(outputs.shape)
            loss = criterion(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Show progress
            if i == len(trainloader): #  or i % 100 == 0 
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, i, len(trainloader), loss.item()))
                
        val_loss = validation(model, valloader, criterion)
            
        early_stopping(val_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            return model, loss
            
        
    return model, loss


def cross_validation(nn_arch, dataset_train, dataset_test, epochs = 50, criterion = nn.L1Loss(), k_folds = 5, batch_size = 32):
    """_summary_

    Args:
        hidden_dims (_type_): _description_
        dataset (_type_): _description_
        epochs (int, optional): _description_. Defaults to 50.
        criterion (_type_, optional): _description_. Defaults to nn.MSELoss().
        k_folds (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    # For fold results
    results =  {'tr': [], 'val': [], 'test': [], 'aic': []}

    
    # early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0.001)

    testloader = torch.utils.data.DataLoader(
                            dataset_test, batch_size=batch_size)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    print('--------------------------------')

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and valing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset_train, 
                          batch_size=batch_size, sampler=train_subsampler, drop_last = True)
        valloader = torch.utils.data.DataLoader(
                          dataset_train,
                          batch_size=batch_size, sampler=val_subsampler, drop_last = True)
        
        

        # Init the neural network
        model = nn_net(dataset_train.x.shape[1],nn_arch).to(device) 
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run the training loop for defined number of epochs
        model, loss = train(model, epochs, optimizer, trainloader, valloader, criterion)
        
        # Early stopping
        val_loss = validation(model, valloader, criterion)
        print('The current loss:', val_loss)

        test_loss = validation(model, testloader, criterion)
        # aic_score = AIC_score(model, len(testloader.dataset), loss.item())
        
        results['tr'].append(loss.item())
        results['val'].append(val_loss)
        results['test'].append(test_loss)
        # results['aic'].append(aic_score)

    return  {'tr': np.mean(results['tr']), 'val': np.mean(results['val']), 'test': np.mean(results['test'])}

def bootstrap(nn_arch, train_data: np.ndarray, test_data: np.ndarray, 
                    epochs = 25, n_bootstraps = 50, bootstrap_dim = 1000, train_fd = False) -> dict:
    """_summary_

    Args:
        nn_arch (_type_): _description_
        train_data (np.ndarray): _description_
        test_data (np.ndarray): _description_
        epochs (int, optional): _description_. Defaults to 25.
        n_bootstraps (int, optional): _description_. Defaults to 50.
        bootstrap_dim (int, optional): _description_. Defaults to 1000.

    Returns:
        dict: _description_
    """
    bootstrap_losses = {'tr': [], 'val': [], 'test': [], 'aic': []}
    # bootstrap trials cycle
    for i in range(n_bootstraps):
        print('--------------------------------')
        print(f'BOOTSTRAP {i}')
        print('--------------------------------')
        # subsample the data
        indexes = np.arange(len(train_data))
        indexes = np.random.choice(indexes, size=bootstrap_dim, replace=True)
        data_bootstrap = nn_dataset(train_data[indexes], train_fd = train_fd)
        data_test = nn_dataset(test_data, train_fd = train_fd)

        # perform cross validation
        cv_losses = cross_validation(nn_arch, data_bootstrap, data_test, epochs, train_fd = train_fd)
        bootstrap_losses['tr'].append(cv_losses['tr'])
        bootstrap_losses['val'].append(cv_losses['val'])
        bootstrap_losses['test'].append(cv_losses['test'])
        bootstrap_losses['aic'].append(cv_losses['aic'])
    bootstrap_losses = {'tr': (np.mean(np.array(bootstrap_losses['tr'])), np.std(np.array(bootstrap_losses['tr']))),
                        'val': (np.mean(np.array(bootstrap_losses['val'])), np.std(np.array(bootstrap_losses['val']))),
                        'test': (np.mean(np.array(bootstrap_losses['test'])), np.std(np.array(bootstrap_losses['test']))),
                        'aic': (np.mean(np.array(bootstrap_losses['aic'])), np.std(np.array(bootstrap_losses['aic'])))
                        }
    return bootstrap_losses


def main(data_train, data_test, nn_arch):
    # data = np.load(filepath, allow_pickle=True)
    
    
    
    # results = bootstrap(nn_arch, data_train, data_test, epochs = 1000, n_bootstraps = 5, bootstrap_dim = 5000)
    train = nn_dataset(data_train)
    test = nn_dataset(data_test)
    cv_losses = cross_validation(nn_arch, train, test, epochs = 1000, batch_size = 64, k_folds = 2)

    return cv_losses