import os
import sys
import numpy as np 
import pandas as pd
import pickle
import math
from tqdm import tqdm, trange
import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 300
import matplotlib as mpl
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn.parameter import Parameter

from datetime import datetime

now = datetime.now() # current date and time

import cmdstanpy

import wandb


class CustomDataset(Dataset):
    def __init__(self, csv_path, input_cols, target_col):
        # Load dataset
        data = pd.read_csv(csv_path)
        self.date_time = pd.to_datetime(data['Datum'])

        # Extract features and target
        self.evalPoints = data[input_cols].values.astype(np.float32)
        self.data = data[target_col].values.astype(np.float32).reshape(-1, 1)  # Reshape target for compatibility

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.evalPoints[idx], self.data[idx]
    
    def get_datetime(self):
        return self.date_time

wandb.init(
    # set the wandb project where this run will be logged
    project="BC Simulation",
    # track hyperparameters and run metadata
    name="M Flow #RTX3070"
)


class PHI(nn.Module):
    '''
    Implementation of feature transformation layer with RBF layer.
    We assume here that alpha is constant for all basis.
    Shape:
        - Input: (N, n_evals, in_features) N is batches
        - Output: (N, n_evals, out_dims), out_dims is a parameter
    Parameters:
        - in_features: number of input dimension for each eval point
        - alpha - trainable parameter controls width. Default is 1.0
        - n_centers - number of points to be used as centers in rbf/matern
        layers. centers are trainable, default is 100
        - hidden_dim1: hidden dimension for 1st layer. Default is 20
        - hidden_dim2: hidden dimension for 2nd layer. Default is 20
        - out_dims: output features to construct. Default is 100
    Examples:
        >>> a1 = PHI(256)
        >>> x = torch.randn(1,256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = 1.0, n_centers = 10, 
                    hidden_dim1 = 20, hidden_dim2 = 20, out_dims = 100):
        '''
        Initialization.
        INPUT:
            - in_features: number of input dimension for each eval point
            - alpha: trainable parameter
            alpha is initialized with 1.0 value by default
            - n_centers: number of points to be used as centers in rbf/matern
            layers. centers are trainable, default is 100
            - hidden_dim1: hidden dimension for 1st layer. Default is 20
            - hidden_dim2: hidden dimension for 2nd layer. Default is 20
            - out_dims: hidden dimension for 2nd layer. Default is 100
        '''
        super(PHI,self).__init__()
        self.in_features = in_features

        # initialize alpha
       # self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        #self.alpha.requiresGrad = True # set requiresGrad to true!
        # centers
        self.centers = Parameter(torch.randn(n_centers, in_features)) # create a tensor out of centers
        self.centers.requiresGrad = True # set requiresGrad to true!
        # linear layers
        self.linear1 = nn.Linear(n_centers, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, out_dims)

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        rbf = torch.exp(-1 * torch.cdist(x, self.centers).pow(2))
        hidden1 = torch.tanh(self.linear1(rbf))
        hidden2 = torch.tanh(self.linear2(hidden1))
        out = self.out(hidden2)
        return out


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.sd = nn.Linear(hidden_dim2, z_dim)
    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        z_mu = self.mu(hidden2)
        # z_mu is of shape [batch_size, z_dim]
        z_sd = self.sd(hidden2)
        # z_sd is of shape [batch_size, z_dim]
        return z_mu, z_sd

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, input_dim)
    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        pred = self.out(hidden2)
        # pred is of shape [batch_size, input_dim]
        return pred


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim1, hidden_dim2, input_dim)

    def reparameterize(self, z_mu, z_sd):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if self.training:
            # sample from the distribution having latent parameters z_mu, z_sd
            # reparameterize
            std = torch.exp(z_sd / 2)
            eps = torch.randn_like(std)
            return (eps.mul(std).add_(z_mu))
        else:
            return z_mu


    def forward(self, x):
        # encode
        z_mu, z_sd = self.encoder(x)
        # reparameterize
        x_sample = self.reparameterize(z_mu, z_sd)
        # decode
        generated_x = self.decoder(x_sample)
        return generated_x, z_mu,z_sd

def calculate_loss_VAE(x, reconstructed_x, mean, log_sd):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_sd - mean.pow(2) - log_sd.exp())
    return RCL + KLD

class PIVAE(nn.Module):
    '''
    Implementation of PIVAE with feature transformation layer (RBF layer).
    Shape:
        - Input: (N, n_evals, in_features) N is batches
        - Output: (N, n_evals, 1), currently we have 1D output only
    Parameters:
        - in_features: number of input dimension for each eval point
        - alpha - trainable parameter controls width. Default is 1.0
        - n_centers - number of points to be used as centers in rbf/matern
        layers. centers are trainable, default is 100
        - dim1: hidden dimension for 1st transformation layer. Default is 20
        - dim2: hidden dimension for 2nd layer. Default is 20
        - out_dims: output features to construct (size of beta and VAE). 
        Default is 100
        - hidden_dim1 - hidden dimensions for 1st layer VAE. Default is 128
        - hidden_dim2 - hidden dimensions for 1st layer VAE. Default is 64
        - z_dim - latent dimension for VAE. Default is 20
        - batch_size - batch_size for training. For now set same as n_samples
    Examples:
        >>> a1 = PHI(256)
        >>> x = torch.randn(1,256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = 1.0, n_centers = 10, dim1 = 20, 
                    dim2 = 20, out_dims = 100, hidden_dim1 = 128, 
                    hidden_dim2 = 64, z_dim = 20, batch_size = 10000):
        super(PIVAE, self).__init__()
        self.out_dims = out_dims
        self.batch_size = batch_size
        self.phi = PHI(in_features, alpha=alpha, n_centers=n_centers, 
                        hidden_dim1=dim1, hidden_dim2=dim2, out_dims=out_dims)
#         self.betas = nn.ModuleList()
#         for _ in range(self.batch_size):
#             self.betas.append(nn.Linear(out_dims, 1))
        self.betas = Parameter(torch.randn(batch_size, out_dims)) # create a beta matrix
        self.betas.requiresGrad = True
        
        self.vae = VAE(input_dim=out_dims, hidden_dim1=hidden_dim1, 
                        hidden_dim2=hidden_dim2, latent_dim=z_dim)
    
    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''        
        phi_x = self.phi(x)
        if phi_x.dim() == 2:  # Reshape if phi_x is 2D
            phi_x = phi_x.unsqueeze(1)
            
        
        y1 = torch.einsum("bo,bno->bn",[self.betas,phi_x])
        
        beta_vae, z_mu, z_sd = self.vae(self.betas)
        
        y2 = torch.einsum("bo,bno->bn",[beta_vae,phi_x])
        
#         ipdb.set_trace()

        return y1, y2, z_mu, z_sd
    
def calculate_loss(target, reconstructed1, reconstructed2, mean, log_var):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed1, target, reduction='sum') + \
                F.mse_loss(reconstructed2, target, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL + KLD

sm = cmdstanpy.CmdStanModel(stan_file='pivae_2d.stan')

def train_piVAE():

    # Step 1: Read the CSV file
    # file_path = 'simulation.csv'
    # data = pd.read_csv(file_path)
    
    # # Step 2: Split the data into training and testing sets
    # # Assuming the target column is named 'target'; adjust as necessary
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
    
    # # Step 3: Save the training and testing sets to separate CSV files
    # train_data.to_csv('train_data.csv', index=False)
    # test_data.to_csv('test_data.csv', index=False)

    directory = os.path.join("train", now.strftime("%m-%d-%Y_%H-%M-%S"))
    

    if not os.path.exists(directory):
        os.makedirs(directory)

    input_cols = ['G_trackerWm', 'G_diffuseWm', 'IR_TrackerWm', 'zenith', 'azimuth', 'T_amb_outdoorC', 'v_wind_speedms', 'T_vent_1C', 'T_IR_Tracker']
    target_col = 'm_flow_1kgh'
    batch_size = 32
  
    # Create data, model, and optimizer
    train_ds = CustomDataset(csv_path='train_simulation.csv', input_cols=input_cols, target_col=target_col)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    
    
    val_ds = CustomDataset(csv_path='test_simulation.csv', input_cols=input_cols, target_col=target_col)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)


    print(len(train_ds))
    print(len(val_ds))

    # Initializing data and model parameters for 2D data
    n_samples = 1000  # Adjust based on your dataset size
    in_features = len(input_cols)
    n_evals = len(val_ds)  
    n_centers = math.ceil(n_evals / 2)
    alpha = 1.0
    dim1 = 128
    dim2 = 128
    hidden_dims1 = 128
    hidden_dims2 = 64
    z_dim = 24
    out_dims = 64
    
    model = PIVAE(in_features=in_features, alpha=alpha, n_centers=n_centers,
                  dim1=dim1, dim2=dim2, out_dims=out_dims, 
                  hidden_dim1=hidden_dims1, hidden_dim2=hidden_dims2, 
                  z_dim=z_dim, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # Check for available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(torch.cuda.current_device()))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model = model.to(device)
    
    
    epochs = 10000  
    print(device)
    best_val_loss = float('inf')
    t = trange(epochs)
    for e in t:
        model.train()
        # Training loop with loss calculation
        total_train_loss = 0
        for features, target in train_dl:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            y1, y2, z_mu, z_sd = model(features)
            loss = calculate_loss(target, y1, y2, z_mu, z_sd)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()
        
        # Average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dl)
        wandb.log({"train_loss": avg_train_loss})
        t.set_description(f'Epoch {e+1} - Train Loss: {avg_train_loss:.3f}')
        
        # Validation loop with full data using Stan
        if e == 0 or e % 200 == 0:  # Run inference every 200 epochs for evaluation

            model.eval()
            phi = model.phi
            vae = model.vae
            vae_decoder = vae.decoder
       
            x_inf = val_ds.evalPoints
            y_inf = val_ds.data.reshape(-1)
            
            ll_idx = np.arange(n_evals) + 1   # Get indices where condition is met
            ll_len = len(ll_idx)

            # Define stan_data for the selected subset
            stan_data = {
                'p': z_dim,
                'p1': hidden_dims1,
                'p2': hidden_dims2,
                'n': n_evals,
                'W1': vae_decoder.linear1.weight.T.cpu().detach().numpy(),
                'B1': vae_decoder.linear1.bias.T.cpu().detach().numpy(),
                'W2': vae_decoder.linear2.weight.T.cpu().detach().numpy(),
                'B2': vae_decoder.linear2.bias.T.cpu().detach().numpy(),
                'W3': vae_decoder.out.weight.T.cpu().detach().numpy(),
                'B3': vae_decoder.out.bias.T.cpu().detach().numpy(),
                'beta_dim': out_dims,
                'phi_x': phi(torch.tensor(x_inf).float().to(device)).cpu().detach().numpy(),
                'y': y_inf,
                'll_len': ll_len,      # New addition for Stan
                'll_idxs': ll_idx       # New addition for Stan
            }
            
            fit = sm.sample(data=stan_data, iter_sampling=n_samples, iter_warmup=200, chains=4, adapt_delta=0.95)
            
            # Get predictions from the Stan model
            out = fit.stan_variables()
            df = pd.DataFrame(out['y2'])

            if e % 1000 == 0:
                path_inference_y = os.path.join(directory, f"mcmc_{e}.csv")
                df.to_csv(path_inference_y, index=False)

            posterior_means = df.mean().to_numpy()

            # R²
            r2 = r2_score(y_inf, posterior_means)
            
            # Pearson Correlation Coefficient
            pearson_corr, _ = pearsonr(y_inf, posterior_means)
            
            # Spearman Rank Correlation Coefficient
            spearman_corr, _ = spearmanr(y_inf, posterior_means)

            wandb.log({"R²": r2})
            wandb.log({"Pearson Correlation Coefficient": pearson_corr})
            wandb.log({"Spearman Rank Correlation Coefficient": spearman_corr})


            x_e = np.random.randint(250, len(val_ds))
            x_s = x_e - 250

            datapoints = val_ds.get_datetime().iloc[x_s:x_e]

            df_sample = df[df.columns[x_s:x_e]]

            fig = plt.figure(figsize=(20, 5))
            ax = fig.add_subplot(111)
            
            # True values as a continuous black line
            ax.plot(datapoints, y_inf[x_s:x_e], color='blue', linewidth=1, label='True')
            
            # 95% Credible Interval as a shaded area
            ax.fill_between(
                datapoints,
                df_sample.quantile(0.025).to_numpy(),
                df_sample.quantile(0.975).to_numpy(),
                facecolor="blue",
                color='blue', 
                alpha=0.2, label='95% Credible Interval'
            )
            
            # Posterior mean as a continuous red line
            ax.plot(datapoints, posterior_means[x_s:x_e], color='red', alpha=0.7, linewidth=1, label='Posterior mean')
            
            # Formatting
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y=f(x)$')
            ax.set_title('Inference fit')
            ax.legend()
            
            # Logging to wandb
            image = wandb.Image(fig)
            wandb.log({"inference fit": image})
                    
        # Validation loss calculation
        total_val_loss = 0
        with torch.no_grad():
            for features, target in val_dl:
                features, target = features.to(device), target.to(device)
                y1, y2, z_mu, z_sd = model(features)
                val_loss = calculate_loss(target, y1, y2, z_mu, z_sd)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dl)
        wandb.log({"val_loss": avg_val_loss})
        t.set_postfix(val_loss=avg_val_loss)
        if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          model_best = "best-model.pth"
          path_model_best = os.path.join(directory, model_best)
          
          torch.save(model, path_model_best)

    model_last = "last-model.pth"
    path_model_last = os.path.join(directory, model_last)
    csv_last = os.path.join(directory, "last-mcmc.csv")
    df.to_csv(csv_last, index=False)
    torch.save(model, path_model_last)
    return model

model = train_piVAE()


