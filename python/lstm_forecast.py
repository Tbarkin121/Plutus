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

#%% min_max_scaler 
def min_max_scaler(tensor, min_val, max_val):
    """
    Scales a tensor to a range [min_val, max_val] and returns scaling parameters.
    
    Args:
        tensor (torch.Tensor): Input tensor to scale.
        min_val (float): Minimum value of the output range.
        max_val (float): Maximum value of the output range.
    
    Returns:
        torch.Tensor: Scaled tensor.
        float: Original minimum value of the tensor.
        float: Original maximum value of the tensor.
    """
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)
    scaled_tensor = (tensor - min_tensor) / (max_tensor - min_tensor) * (max_val - min_val) + min_val
    return scaled_tensor, min_tensor, max_tensor

# Example usage:
features = torch.randn(10, 5)  # Random tensor simulating features
scaled_features, min_tensor, max_tensor = min_max_scaler(features, 0, 1)

# Function to reverse the Min-Max scaling
def inverse_min_max_scaler(scaled_tensor, min_tensor, max_tensor, min_val, max_val):
    return (scaled_tensor - min_val) / (max_val - min_val) * (max_tensor - min_tensor) + min_tensor

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

#%% just checking the inverse scaler function real quick
# Example to reverse scale the 'AAPL_open_scaled' column
aapl_open_original = inverse_standard_scaler(torch.tensor(df['AAPL_open_scaled']), torch.tensor(scale_params['AAPL_open']['mean']), torch.tensor(scale_params['AAPL_open']['std']))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['AAPL_open'], label='Original AAPL Open', linestyle='-', marker='o')
plt.plot(aapl_open_original, label='Inverse Scaled AAPL Open', linestyle='--', marker='x')
plt.title('Comparison of Original and Inverse Scaled AAPL Open Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


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
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        outputs, (hn, cn) = self.lstm(x, (h0, c0))
        # print('model forward ----')
        # print(outputs.shape)
        # print(outputs[:, -1, :].shape)
        out = self.fc(outputs[:, -1, :])
        # print(out.shape)
        return out

# def init_weights(m):
#     if isinstance(m, nn.LSTM):
#         for name, param in m.named_parameters():
#             if 'weight_ih' in name:
#                 torch.nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 torch.nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
#     elif isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         m.bias.data.fill_(0)




input_dim = 10  # Number of features (Currently set to just the open_scaled values)
hidden_dim = 128  # Number of LSTM units
num_layers = 20  # Input Sequence Length (Number of time steps)
output_dim = 20  # Output Sequence Length (Number of time steps)

model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim)
# model.apply(init_weights)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


#%% create dataset 
# Assuming df is your loaded DataFrame
stock_dataset = StockDataset(df, sequence_length=num_layers, target_length=output_dim)
# Create a DataLoader
batch_size = 1024  # Adjust batch size according to your needs
train_loader = DataLoader(dataset=stock_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=stock_dataset, batch_size=len(df), shuffle=True)
   
    
#%%
def plot_gradients(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color='b')
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color='k' )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
    
#%%
num_epochs = 5  # Number of epochs to train
clip_value = 0.1  # Gradient norm threshold

model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    i = 0
    for features, targets in train_loader:
        # Move data to the appropriate device
        features = features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        
        # Apply gradient clipping; clip_value is a hyperparameter and can be adjusted.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
       
        # plot_gradients(model)
        optimizer.step()  # Update parameters
        
        if(i%10==0):
            print(f'Loss {i}: {loss.item():.4f}')
        i += 1
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

