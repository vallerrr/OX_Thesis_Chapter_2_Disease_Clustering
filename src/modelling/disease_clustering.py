"""
# Created by valler at 23/06/2024
Feature: 

"""


from src import params
from src.data_preprocess import utils
from collections import Counter
import numpy as np
import ast
import torch
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
# =================================================================================================
# 0. prepare the data
# =================================================================================================
# 0.1 read the basic columns
record_column = params.disease_record_column
dates_col, df_single_record = utils.import_df_single_record(record_column)
level = 'chronic'
nan_str = params.nan_str
cols = [x for x in df_single_record.columns if 'phe' in x]
selected_cols = ['diseases_within_window_phecode_selected_chronic','diseases_within_window_phecode_selected_category_chronic']

# with gender and age? yes as they are part of the physical conditions
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')
df_phe_selected = df_single_record[['eid','21022','31']+selected_cols].copy()

# 0.2 add more columns to describe the diseases within the window
for column in selected_cols:
    df_phe_selected[column] = [ast.literal_eval(x) if str(x) not in nan_str else None for x in df_phe_selected[column]]
df_phe_selected['diseases_count'] = [len(x) if str(x) not in nan_str else 0 for x in df_phe_selected['diseases_within_window_phecode_selected_chronic']]


# 0.3 flatten the selected_cols
phe_unique = df_phe_selected[selected_cols[0]].explode().explode().dropna().unique().tolist()
cat_unique = df_phe_selected[selected_cols[1]].explode().explode().dropna().unique().tolist()

# phe code level
df_phe_selected['phe_count'] = [Counter(x) if str(x) not in nan_str else None for x in df_phe_selected[selected_cols[0]]]
for column in phe_unique:
    df_phe_selected[f'd_{column}'] = [None if (str(x) == 'None') else x[column] if (column in x.keys()) else None for x in df_phe_selected['phe_count']]


# phe cat level
df_phe_selected['cat_count'] = [Counter(x) if str(x) not in nan_str else None for x in df_phe_selected[selected_cols[1]]]
for column in cat_unique:
    df_phe_selected[f'c_{column}'] = [None if(str(x)=='None') else x[column] if (column in x.keys()) else None for x in df_phe_selected['cat_count']]

df_phe_selected.drop(columns=['phe_count','cat_count']+selected_cols,inplace=True)
df_phe_selected.fillna(0, inplace=True)

# =================================================================================================
# 1. Disease HDBSCAN clustering
"""
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise): 
An extension of DBSCAN that handles varying densities better than DBSCAN. 
It is very effective for large datasets where cluster density varies.

@ 25 June 2024, this method does not return solid cluster results.
maybe turn to use Deep Learning Methods 
"""
# =================================================================================================

import hdbscan
from sklearn.preprocessing import StandardScaler
hdb = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True, min_samples=100000)
scaler = StandardScaler()

data = df_phe_selected.drop(columns=['eid']).drop(df_phe_selected.loc[df_phe_selected['diseases_count']==0].index)
X_scaled = scaler.fit_transform(data)
hdb.fit(X_scaled)

labels = hdb.labels_

probabilities = hdb.probabilities_

# Access the outlier scores
outlier_scores = hdb.outlier_scores_

print("Cluster labels:", labels)
print("Probabilities of cluster membership:", probabilities)
print("Outlier scores:", outlier_scores)
from datetime import datetime
print(datetime.now())



# =================================================================================================




# =================================================================================================
# 2. Diseases VAE clustering 
# =================================================================================================

import torch
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets


mps_device = torch.device("mps")
torch.set_default_device(mps_device)

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Convert to tensor if necessary
        return torch.tensor(row.values, dtype=torch.float32)


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=mps_device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mps_device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar




num_epochs=50
batch_size = 1000
dataset = CustomDataset(df_phe_selected.drop(columns=['eid']))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train(model, optimizer, epochs, device,train_loader, batch_size):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.view(batch_size, -1).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
    return overall_loss

model = VAE().to(mps_device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train(model=model, optimizer=optimizer, epochs=10, device=mps_device,train_loader=dataloader,batch_size=batch_size)


