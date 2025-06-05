# -*- coding: utf-8 -*-
"""
@author: SinaJahangir
# Multi-timescale LSTM for daily and weekly forecasting. The model is designed
to use Daymet and ERA5 data
# This model is benchmarked against the HDL method
"""
# Import Pytorch
import torch.nn as nn
import torch
#%%

class LSTMModel(nn.Module):
    """
    LSTM model with two inputs processed by separate LSTM layers, 
    with two outputs (1D and 7D) from the encoded spaces.
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, num_layers=1,dropout_prob = 0.4):
        """
        Args:
            input_dim1 (int): Number of features for the first input.
            input_dim2 (int): Number of features for the second input.
            hidden_dim (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModel, self).__init__()
        
        # LSTM layers for the two inputs
        self.lstm1 = nn.LSTM(input_dim1, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim, num_layers, batch_first=True)
        # Dropout layer for avoiding overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers for 1D and 7D outputs
        self.fc_1d = nn.Linear(hidden_dim+hidden_dim, 1)  # For 1D output
        self.fc_7d = nn.Linear(hidden_dim+hidden_dim, 7)  # For 7D output

    def forward(self, input1, input2):
        """
        Forward pass through the model.
        
        Args:
            input1 (torch.Tensor): First input of shape [batch_size, seq_length, input_dim1].
            input2 (torch.Tensor): Second input of shape [batch_size, seq_length, input_dim2].
        
        Returns:
            torch.Tensor: 1D output of shape [batch_size, 1].
            torch.Tensor: 7D output of shape [batch_size, 7].
        """
        # Process inputs through separate LSTMs
        lstm_out_daymet, _ = self.lstm1(input1)  # h_n1: [num_layers, batch_size, hidden_dim]
        lstm_out_era5, _ = self.lstm2(input2)  # h_n2: [num_layers, batch_size, hidden_dim]
        
        # Extract the last hidden states (output at the final time step)
        last_out_daymet = lstm_out_daymet[:, -1, :]  # [batch_size, hidden_dim]
        last_out_era5 = lstm_out_era5[:, -1, :]      # [batch_size, hidden_dim]
        
        # Concatenate the last hidden states along the feature dimension
        combined_features = torch.cat((last_out_daymet, last_out_era5), dim=1)
        combined_features= self.dropout(combined_features)
        # Pass through separate linear layers for 1D and 7D outputs
        output_1d = self.fc_1d(combined_features)  # [batch_size, 1]
        output_7d = self.fc_7d(combined_features)  # [batch_size, 7]
        return output_7d, output_1d