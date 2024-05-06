# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:34:17 2024

@author: Plutonium
"""


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.cuda.is_available()
# torch.set_default_device('cuda')  #Interfearing with the DataLoader(..., Shuffle=True)
#%%

# Load from CSV
df = pd.read_csv('dict_vol_df_combined.csv')


#%% standard_scaler
def standard_scaler(tensor):
    """
    Scales a tensor to have zero mean and unit variance, and returns scaling parameters.
    
    Args:
        tensor (torch.Tensor): Input tensor to standardize.
    
    Returns:
        torch.Tensor: Standardized tensor.
        float: Mean of the tensor before scaling.
        float: Standard deviation of the tensor before scaling.
    """
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor, mean, std

# Example usage:
features = torch.randn(10, 5)  # Random tensor simulating features
standardized_features, mean, std = standard_scaler(features)

# Function to reverse the standard scaling
def inverse_standard_scaler(standardized_tensor, mean, std):
    return standardized_tensor * std + mean

#%% updating dictionary
# Scaling selected columns and storing parameters
scale_params = {}
for column in df.columns:
    if column.endswith('_open') or column.endswith('_close'):
        tensor = torch.tensor(df[column].values, dtype=torch.float32)
        scaled_tensor, mean, std = standard_scaler(tensor)
        df[column + '_scaled'] = scaled_tensor.numpy()  # Create new column for scaled data
        scale_params[column] = {'mean': mean.item(), 'std': std.item()}

# Display the updated DataFrame and scaling parameters
print(df.head())
print(scale_params)



#%% define dataset 
class StockDataset(Dataset):
    def __init__(self, dataframe, sequence_length=50, target_length=1):
        """
        Args:
            dataframe (DataFrame): Pandas DataFrame containing the stock data.
            sequence_length (int): The number of time steps to be used as input features.
            target_length (int): The number of time steps to predict.
        """
        self.sequence_length = sequence_length
        self.target_length = target_length
        self.features = dataframe.filter(like='_open_scaled').values  # Example: using only '_open' columns
        self.targets = dataframe['AAPL_close_scaled'].values  # Example: predicting 'AAPL_close'
    
    def __len__(self):
        return len(self.features) - self.sequence_length - self.target_length + 1

    def __getitem__(self, index):
        return (self.features[index:index+self.sequence_length],
                self.targets[index+self.sequence_length:index+self.sequence_length+self.target_length])



#%% define network model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)  # To match input dimension to model dimension
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_dim)
        src = self.input_linear(src)  # shape: (seq_length, batch_size, d_model)
        output = self.transformer_encoder(src)  # shape: (seq_length, batch_size, d_model)
        output = self.output_linear(output)  # shape: (seq_length, batch_size, output_dim)
        return output


sequence_length = 50  # This is a typical sequence length for time series forecasting

input_dim = 10  # Number of features in the input
d_model = 512  # Dimensionality of the model
nhead = 8  # Number of heads in the multiheadattention models
num_encoder_layers = 6  # Number of sub-encoder-layers in the encoder
dim_feedforward = 2048  # Dimensionality of the feedforward network model
output_dim = 10  # Number of features in the output

model = TimeSeriesTransformer(
    input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


#%% create dataset 
# Assuming df is your loaded DataFrame
stock_dataset = StockDataset(df, sequence_length=sequence_length, target_length=output_dim)
# Create a DataLoader
batch_size = 1024  # Adjust batch size according to your needs
train_loader = DataLoader(dataset=stock_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=stock_dataset, batch_size=len(df), shuffle=True)
   

#%%
num_epochs = 5  # Number of epochs to train

model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    for features, targets in train_loader:
        features = features.permute(1, 0, 2).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Transpose for Transformer (seq_length, batch, features)
        targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#%%
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for features, targets in test_loader:
        features = features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        outputs = model(features)
        
        targets_unscaled = inverse_standard_scaler(targets, torch.tensor(scale_params['AAPL_open']['mean']), torch.tensor(scale_params['AAPL_open']['std']))
        outputs_unscaled = inverse_standard_scaler(outputs, torch.tensor(scale_params['AAPL_open']['mean']), torch.tensor(scale_params['AAPL_open']['std']))
        loss = criterion(outputs_unscaled, targets_unscaled)
        print(loss)
    
#%%

plt.close('all')

targets_np = targets_unscaled.cpu().detach().numpy()
outputs_np = outputs_unscaled.cpu().detach().numpy()


# Iterate over first few samples
for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(targets_np[i, :], label=f'Original AAPL Open {i}', linestyle='-', marker='o')
    plt.plot(outputs_np[i, :], label=f'Predicted AAPL Open {i}', linestyle='--', marker='x')

    plt.title('Comparison of Original and Predicted AAPL Open Prices')
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
#%%
# Feature engineering
# df['return'] = df['close'].pct_change()  # Create returns feature

# Data normalization
# scaler = MinMaxScaler(feature_range=(0, 1))
# df['normalized_return'] = scaler.fit_transform(df[['return']])

