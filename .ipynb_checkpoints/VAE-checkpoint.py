import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import plot_util
from tqdm.notebook import tqdm
import math
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VariationalAutoencoder(nn.Module):
    """
    Class implementing the Variational AutoEncoder
    """
    def __init__(self, hyper_params, is_mnist=True):
        """
        initialize the model
        :param hyper_params: hyperparameters
        :param is_mnist: is the dataset mnist
        """
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(hyper_params, is_mnist)
        self.decoder = Decoder(hyper_params, is_mnist)
        self.is_mnist = is_mnist
        self.hyper_params = hyper_params

    def forward(self, x):
        """
        perform forward pass for VAE. Call forward of encoder then decoder.
        :param x: input data
        """
        z, mean, log_var = self.encoder(x)
        return self.decoder(z), mean, log_var

class VariationalEncoder(nn.Module):
     """
    Class implementing the Variational Encoder
    """
    def __init__(self, hyper_params, is_mnist=True):
        """
        initialize the Variational Encoder
        :param hyper_params: hyperparameters
        :param is_mnist: is the dataset mnist
        """
        super(VariationalEncoder, self).__init__()
        
        if not is_mnist:
            self.hidden = nn.Sequential(
                nn.Linear(hyper_params['input_dims'], hyper_params['hidden_dims']),
                nn.ReLU(),
#                 nn.Linear(256, 256),
#                 nn.ReLU())
            )
        else:
            self.hidden = nn.Sequential(
                nn.Linear(hyper_params['input_dims'], hyper_params['hidden_dims']),
                nn.ReLU(),
                nn.Linear(hyper_params['hidden_dims'], hyper_params['hidden_dims']),
                nn.ReLU()
            )
            
        self.mean_layer = nn.Linear(hyper_params['hidden_dims'], hyper_params['latent_dims'])
        
        self.log_var_layer = nn.Sequential(
            nn.Linear(hyper_params['hidden_dims'], hyper_params['latent_dims']))
        
        self.is_mnist = is_mnist
    
    def forward(self, x):
        """
        perform forward pass for encoder
        :param x: input data
        """
        x = torch.flatten(x, start_dim=1)
        x = self.hidden(x)

        mu =  self.mean_layer(x)
        
        log_var = self.log_var_layer(x)

        # generate new data point in z
        std = torch.exp( 0.5*log_var)
        z = get_point_from_diagonal_normal(mu, std)
        return z, mu, log_var
        
class Decoder(nn.Module):
     """
    Class implementing the Decoder
    """
    def __init__(self, hyper_params, is_mnist=True):
        """
        initialize the Decoder
        :param hyper_params: hyperparameters
        :param is_mnist: is the dataset mnist
        """
        super(Decoder, self).__init__()
        if is_mnist:
            self.hidden = nn.Sequential(
                nn.Linear(hyper_params['latent_dims'], hyper_params['hidden_dims']),
                nn.ReLU(),
                nn.Linear(hyper_params['hidden_dims'], hyper_params['input_dims']),
                nn.Sigmoid()  # output is between 0 and 1
            )
        else:
            self.hidden = nn.Sequential(
                nn.Linear(hyper_params['latent_dims'], hyper_params['hidden_dims']),
                nn.ReLU(),
                nn.Linear(hyper_params['hidden_dims'], hyper_params['input_dims'])
#                 nn.Sigmoid()  # output is between 0 and 1
            )
#             784
        self.is_mnist = is_mnist
    
        if not hyper_params['use_BCE']:
            self.log_var_recons = nn.Parameter(torch.tensor(1.))
        
    def forward(self, z):
        """
        perform forward pass for decoder
        :param x: input data
        """
        z = self.hidden(z)
        
        if self.is_mnist:
            return z.reshape((-1, 1, 28, 28))
        
        return z
    
def get_point_from_diagonal_normal(mean, std):
    """
    sample a point from diagonal normal distribution
    :param mean: mean of distribution
    :param std: standard deviation
    """
    base_point = torch.rand_like(mean)
    return std * base_point + mean


def loss_function(x, x_hat, mean, log_var, autoencoder, recons_weight=1):
    """
    Calculate total loss
    :param x: original data
    :param std: reconstructed data
    :param log_var: log of variance
    :param autoencoder: the autoencoder
    :param recons_weight: weight for reconstruction loss
    """
    if not autoencoder.hyper_params['use_BCE']:
        logvar_recons = autoencoder.decoder.log_var_recons * torch.ones_like(x_hat)
        recons_loss = -torch.sum(0.5 * (logvar_recons + (x - x_hat) ** 2 / (torch.exp(logvar_recons)) + math.log(2 * np.pi)))
    else:
        recons_loss = -F.binary_cross_entropy(x_hat, x, reduction='sum')
        
    kl_loss = 0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - 0.5 * log_var - 1)
#     print(recons_loss, kl_loss)
    return recons_weight * recons_loss - kl_loss

def test(vae, test_dataloader, epochs=100):
    """
    Perform testing on test data
    :param vae: the autoencoder
    :param test_dataloader: test dataloader
    :param epochs: no of epochs
    """
    loss_list = []
    no_data_points=0

    vae.eval()

    with torch.no_grad():
        for epoch in tqdm(range(epochs), total=len(range(epochs))):
                loss_sum=0

                for data in test_dataloader:
                    if vae.is_mnist:
                        x,_ = data
                    else:
                        x = data
                    x = x.to(device) # GPU
                    no_data_points += x.size(0)
                    x_hat, mean, log_var = vae(x)
                    loss = -loss_function(x, x_hat, mean, log_var, vae, 1) / x.shape[0]
                    loss_sum += loss.item()
#                 print('Epoch', epoch, 'test loss', loss_sum/no_data_points)
                loss_list.append(loss_sum/no_data_points)

    plot_util.plot_loss(loss_list)
    
def train(autoencoder, train_dataloader, test_dataloader=None, make_plots=True, epochs=100):
    """
    Perform training on train data
    :param autoencoder: the autoencoder
    :param train_dataloader: train dataloader
    :param test_dataloader: test dataloader
    :param make_plots: make plots boolean
    :param epochs: number of epochs
    """
    autoencoder.train()
    
    opt = torch.optim.Adam(autoencoder.parameters(),lr=autoencoder.hyper_params['lr'])
    
    if autoencoder.is_mnist:
        digits, num = next(iter(test_dataloader))
        
    recons_digit_data = []
        
    for epoch in tqdm(range(epochs)):
        
        loss_sum=0
        no_data_points=0
                
        for data in train_dataloader:
            if autoencoder.is_mnist:
                x,_ = data
            else:
                x = data
            x = x.to(device) # GPU
            no_data_points += x.size(0)
            x_hat, mean, log_var = autoencoder(x)
            loss = -loss_function(x, x_hat, mean, log_var, autoencoder, 1) / x.shape[0]
            loss_sum += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch', epoch, 'train loss', loss_sum/no_data_points)
        
        if autoencoder.is_mnist:
            
            if epoch in [0,4,24,49,99]:

                reconstructed_digits,_,_ = autoencoder(digits)
                reconstructed_digits = reconstructed_digits.detach().numpy()
                recons_digit_data.append(reconstructed_digits)

                if make_plots:
                    plt.figure('decoded_prior_{}'.format(epoch), figsize=(5, 5))
                    plt.title('Samples From Prior. Epoch : {}'.format(epoch))
                    plot_util.plot_reconstructed(autoencoder, r0=(-3, 3), r1=(-3, 3))

#                 if make_plots:
                    plt.figure('latent_space_{}'.format(epoch), figsize=(5, 5))
                    plt.title('Latent Space. Epoch : {}'.format(epoch))
                    plot_util.plot_latent(autoencoder, train_dataloader)
     
    if autoencoder.is_mnist:
        plt.figure('latent_space_recons')
        plot_util.plot_recons_digit_data(recons_digit_data, test_dataloader)
    
    return autoencoder