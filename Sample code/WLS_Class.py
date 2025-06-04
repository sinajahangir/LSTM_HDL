# -*- coding: utf-8 -*-
"""
@author: SinaJahangir
# WLS_D (WLS for brevity) reconciliation class implemenetd for Pytorch
# The wieghts of the network should be optimized  
"""
# Import Pytorch
import torch.nn as nn
import torch
#%%
# Set GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# WLS class has been devloped for weekly reconciliation. You can change this based
# on your own hierarchy structure
class WLS(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16):
        """Initializes the instance attributes"""
        super(WLS, self).__init__()
        
        # Initialize S matrix
        s_temp = torch.diag(torch.ones(7).to(device))
        s_temp = torch.cat((torch.ones(1, 7).to(device), s_temp), dim=0)
        self.S = torch.tensor(s_temp, dtype=torch.float32, device=device)  # [8, 7]

        # MLP to predict sigma diagonal elements
        self.sigma_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Ensure sigma values remain positive
        )

    def forward(self, inputs):
        """
        Defines the computation from inputs to outputs
        :param inputs: torch.Tensor of shape [batch_size, 8]
        :return: reconciled output of shape [batch_size, 8]
        """

        # Predict diagonal sigma elements
        sigma_diag = self.sigma_mlp(inputs)  # [batch_size, 8]
        sigma_diag = torch.diag_embed(sigma_diag)  # Convert to diagonal matrix [batch_size, 8, 8]

        # Compute sigma inverse with small epsilon for stability
        epsilon = 1e-6
        sigma_inverse = torch.linalg.inv(sigma_diag + torch.eye(8, device=device).unsqueeze(0) * epsilon)  # [batch_size, 8, 8]

        # Compute (Sᵀ * sigma⁻¹ * S)⁻¹
        S_t = self.S.T  # [7, 8]
        S_t_S = torch.matmul(S_t, torch.matmul(sigma_inverse, self.S))  # [batch_size, 7, 7]
        S_t_S_inv = torch.linalg.inv(S_t_S)  # [batch_size, 7, 7]

        # Compute beta = (Sᵀ * sigma⁻¹ * S)⁻¹ * Sᵀ * sigma⁻¹ * inputs
        inputs_expanded = inputs.unsqueeze(-1)  # [batch_size, 8, 1]
        beta = torch.matmul(S_t_S_inv, torch.matmul(S_t, torch.matmul(sigma_inverse, inputs_expanded)))  # [batch_size, 7, 1]

        # Reconcile the outputs: reconcile = S * beta
        reconcile = torch.matmul(self.S, beta)  # [batch_size, 8, 1]

        # Squeeze the last dimension to match input shape
        reconcile = reconcile.squeeze(-1)  # [batch_size, 8]

        return reconcile

