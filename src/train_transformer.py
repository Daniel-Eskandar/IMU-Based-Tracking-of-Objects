import torch
from torch import nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle

def load_dataset(data_path):
    source_columns = ["gx(rad/s)", "gy(rad/s)", "gz(rad/s)", "ax(m/s^2)", "ay(m/s^2)", "az(m/s^2)"]
    target_columns = ["px", "py", "pz"]
    
    source_sequences = []
    target_sequences = []
    
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
        
            # Read the file using pandas
            df = pd.read_csv(file_path, sep=" ")
            
            # Extract required columns
            source_data = df[source_columns]
            target_data = df[target_columns]
        
            # Create source sequences of size (100, 6)
            for i in range(0, len(source_data) - 50, 50):
                source_seq = source_data.iloc[i:i+100, :].values
                source_sequences.append(source_seq)
        
            # Create target sequences of size (100, 3)
            for i in range(0, len(target_data) - 50, 50):
                target_seq = target_data.iloc[i:i+100, :].values
                target_sequences.append(target_seq)
        
            # Replace the last source sequence with the last 100 entries of the file
            last_source_seq = source_data.iloc[-100:, :].values
            source_sequences[-1] = last_source_seq
    
            # Replace the last target sequence with the last 100 entries of the file
            last_target_seq = target_data.iloc[-100:, :].values
            target_sequences[-1] = last_target_seq
    
    # Subtract the first row from all rows in each target sequence
    target_sequences = [seq - seq[0] for seq in target_sequences]
    
    # Convert source sequences to torch tensor
    source_tensors = torch.stack([torch.from_numpy(seq) for seq in source_sequences]).to(torch.float32)
    
    # Convert target sequences to torch tensor
    target_tensors = torch.stack([torch.from_numpy(seq) for seq in target_sequences]).to(torch.float32)
    
    # Create TensorDatasets
    dataset = TensorDataset(source_tensors, target_tensors)
    # target_dataset = TensorDataset(target_tensors)

    return dataset



class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        input_dim,
        output_dim,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding_replacement_input = nn.Linear(input_dim, dim_model)
        self.embedding_replacement_output = nn.Linear(output_dim, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, output_dim)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length, input_dim)
        # Tgt size must be (batch_size, tgt sequence length, output_dim)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # print(src.shape, tgt.shape)
        src = self.embedding_replacement_input(src) * math.sqrt(self.dim_model)
        tgt = self.embedding_replacement_output(tgt) * math.sqrt(self.dim_model)
        # print(src.shape, tgt.shape)
        src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)
        # print(src.shape, tgt.shape)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask


def train_loop(model, opt, loss_fn, dataloader, device=torch.device('cpu')):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[0].to(device), batch[1].to(device)

        y_input = torch.zeros(y.shape).to(device)
        y_expected = y
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1,0,2)     
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.cpu().detach().item()
        
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader, device=torch.device('cpu')):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)

            y_input = torch.zeros(y.shape).to(device)
            y_expected = y

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1,0,2)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.cpu().detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, device=torch.device('cpu')):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device=device)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader, device=device)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list
    


def main():
    train_data_path = './../dat/merged/train'
    train_dataset = load_dataset(train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_data_path = './../dat/merged/val'
    val_dataset = load_dataset(val_data_path)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_data_path = './../dat/merged/test'
    test_dataset = load_dataset(test_data_path)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Transformer(
        input_dim=6, output_dim=3, dim_model=128, num_heads=4, num_encoder_layers=4, num_decoder_layers=4, dropout_p=0.1
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.MSELoss()

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_loader, val_loader, 30, device=device)

    with open('train_loss_list.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open('val_loss_list.pkl', 'wb') as f:
        pickle.dump(validation_loss_list, f)

    torch.save(model, "final_model.pt")


if __name__=="__main__":
    main()