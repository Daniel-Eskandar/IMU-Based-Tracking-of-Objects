# Example Run
# python train_lstm.py --input_size 6 --hidden_size 256 --output_size 3 --num_layers 6 --batch_size 16 --data_path ./../dat/merged --num_epochs 500

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add argparse for command line hyperparameters
parser = argparse.ArgumentParser(description='LSTM Training Script')
parser.add_argument('--data_path', type=str, default='./../dat/merged', help='Path to the directory containing train, val, and test data')
parser.add_argument('--input_size', type=int, default=6, help='Input size of the LSTM')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the LSTM')
parser.add_argument('--output_size', type=int, default=3, help='Output size of the LSTM')
parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the LSTM')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

# Function to load dataset
def load_dataset(data_path):
    source_columns = ["gx(rad/s)", "gy(rad/s)", "gz(rad/s)", "ax(m/s^2)", "ay(m/s^2)", "az(m/s^2)"]
    target_columns = ["px", "py", "pz"]
    source_sequences = []
    target_sequences = []
    
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
            df = pd.read_csv(file_path, sep=" ")
            source_data = df[source_columns]
            target_data = df[target_columns]
        
            for i in range(0, len(source_data) - 50, 50):
                source_seq = source_data.iloc[i:i+100, :].values
                source_sequences.append(source_seq)
        
            for i in range(0, len(target_data) - 50, 50):
                target_seq = target_data.iloc[i:i+100, :].values
                target_sequences.append(target_seq)
        
            last_source_seq = source_data.iloc[-100:, :].values
            source_sequences[-1] = last_source_seq
    
            last_target_seq = target_data.iloc[-100:, :].values
            target_sequences[-1] = last_target_seq

    # Subtract the first row from all rows in each target sequence
    # target_sequences = [seq - seq[0] for seq in target_sequences]
    
    source_tensors = torch.stack([torch.from_numpy(seq) for seq in source_sequences])
    target_tensors = torch.stack([torch.from_numpy(seq) for seq in target_sequences])
    
    source_dataset = TensorDataset(source_tensors)
    target_dataset = TensorDataset(target_tensors)

    return source_dataset, target_dataset

# Load datasets
train_data_path = os.path.join(args.data_path, 'train')
train_dataset = load_dataset(train_data_path)

val_data_path = os.path.join(args.data_path, 'val')
val_dataset = load_dataset(val_data_path)

test_data_path = os.path.join(args.data_path, 'test')
test_dataset = load_dataset(test_data_path)

# Print the shapes of train, validation, and test datasets
print()
print("Train Dataset Source Shape:", train_dataset[0].tensors[0].shape)
print("Train Dataset Target Shape:", train_dataset[1].tensors[0].shape)
print("Validation Dataset Shape:", val_dataset[0].tensors[0].shape)
print("Validation Dataset Target Shape:", val_dataset[1].tensors[0].shape)
print("Test Dataset Shape:", test_dataset[0].tensors[0].shape)
print("Test Dataset Target Shape:", test_dataset[1].tensors[0].shape)
print()

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=6):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

# Convert datasets to PyTorch DataLoader
train_loader = DataLoader(TensorDataset(train_dataset[0].tensors[0].float(), 
                                       train_dataset[1].tensors[0].float()), 
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_dataset[0].tensors[0].float(), 
                                     val_dataset[1].tensors[0].float()), 
                        batch_size=args.batch_size)
test_loader = DataLoader(TensorDataset(test_dataset[0].tensors[0].float(), 
                                      test_dataset[1].tensors[0].float()), 
                         batch_size=args.batch_size)

# Define the LSTM model
model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, 
             output_size=args.output_size, num_layers=args.num_layers)

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
train_losses = []
val_losses = []

for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/ {args.num_epochs}'):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in tqdm(val_loader, desc=f'Validation'):
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch {epoch+1} / {args.num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')

# Plotting the losses
fig, ax = plt.subplots()

ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")

ax.set_xlim([0, len(train_losses) + 1])
ax.set_ylim([0, max(max(train_losses), max(val_losses) + 0.1)])

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

ax.legend()
plt.show()

# Testing
print()
model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_inputs, test_targets in tqdm(test_loader, desc='Testing'):
        test_outputs = model(test_inputs)
        test_loss += criterion(test_outputs, test_targets).item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}')
print()