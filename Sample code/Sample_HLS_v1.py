# -*- coding: utf-8 -*-
"""
@author: SinaJahangir
# This script processes test data for multiple catchments, evaluates a trained HDL (HLS) model,  
and saves the results as CSV files for each catchments.  
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

# Set working directory (consider avoiding global directory changes)
BASE_DIR = Path(r"F:\msjahang\HDL")
#%%
# File paths
#Daymet and ERA5 data used for model development
TRAIN_FILE = BASE_DIR / "All421data_Train_st.csv"
TEST_FILE = BASE_DIR / "All421data_Test_st.csv"
TRAIN_ERA_FILE = BASE_DIR / "All421data_Train_st_ERA5_v2.csv"
TEST_ERA_FILE = BASE_DIR / "All421data_Test_st_ERA5_v2.csv"


def load_and_clean_data(csv_path: Path, dropna: bool = True) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame and optionally removes NaN values.
    """
    df = pd.read_csv(csv_path)
    return df.dropna() if dropna else df


def compute_normalization_stats(df: pd.DataFrame, exclude_cols=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the mean and standard deviation for normalization, excluding specified columns.
    """
    if exclude_cols is None:
        exclude_cols = []
    return np.asarray(df.drop(columns=exclude_cols).mean()), np.asarray(df.drop(columns=exclude_cols).std())


def normalize_dataframe(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, exclude_cols=None) -> pd.DataFrame:
    """
    Normalizes the DataFrame using provided mean and std, preserving excluded columns.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    df_norm = df.copy()
    df_norm.iloc[:, 1:] = (df.iloc[:, 1:] - mean) / std  # Apply normalization
    df_norm[exclude_cols] = df[exclude_cols]  # Restore excluded columns
    return df_norm


# Load datasets
df_train = load_and_clean_data(TRAIN_FILE)
df_test = load_and_clean_data(TEST_FILE, dropna=False)
df_train_era = load_and_clean_data(TRAIN_ERA_FILE)
df_test_era = load_and_clean_data(TEST_ERA_FILE, dropna=False)

# Compute normalization stats (excluding 'basin_id' from normalization)
mean_train, std_train = compute_normalization_stats(df_train, exclude_cols=['basin_id'])
mean_train_era, std_train_era = compute_normalization_stats(df_train_era, exclude_cols=['basin_id'])

# Normalize datasets
df_train_norm = normalize_dataframe(df_train, mean_train, std_train, exclude_cols=['basin_id'])
df_test_norm = normalize_dataframe(df_test, mean_train, std_train, exclude_cols=['basin_id'])
df_train_era_norm = normalize_dataframe(df_train_era, mean_train_era, std_train_era, exclude_cols=['basin_id'])
df_test_era_norm = normalize_dataframe(df_test_era, mean_train_era, std_train_era, exclude_cols=['basin_id'])

# Mean and standard deviation for 'q' column
mean_q = df_train['q'].mean()
std_q = df_train['q'].std()

# Display results
print(f"Training data normalized: {df_train_norm.shape}")
print(f"Test data normalized: {df_test_norm.shape}")
print(f"ERA5 training data normalized: {df_train_era_norm.shape}")
print(f"ERA5 test data normalized: {df_test_era_norm.shape}")
print(f"Mean Q: {mean_q}, Std Q: {std_q}")
#%%
#Error metrics
def nash_sutcliffe_error(Q_obs,Q_sim):
    """
    Written by: SJ
    Q_obs: observed discharge; 1D vector
    Q_sim: simulated discharge; 1D vector
    This function calculates the NSE between observed and simulated discharges
    returns: NSE; float
    """
    if len(Q_sim)!=len(Q_obs):
        print('Length of simulated and observed discharges do not match')
        return
    else:
        num=np.sum(np.square(Q_sim-Q_obs))
        den=np.sum(np.square(Q_obs-np.mean(Q_obs)))
        NSE=1-(num/den)
        return NSE

def CC(Pr,Y):
    from scipy import stats
    Pr=np.reshape(Pr,(-1,1))
    Y=np.reshape(Y,(-1,1))
    return stats.pearsonr(Pr.flatten(),Y.flatten())[0]
def KGE(prediction,observation):

    nas = np.logical_or(np.isnan(prediction), np.isnan(observation))
    pred=np.copy(np.reshape(prediction,(-1,1)))
    obs=np.copy(np.reshape(observation,(-1,1)))
    r=CC(pred[~nas],obs[~nas])
    beta=np.nanmean(pred)/np.nanmean(obs)
    gamma=(np.nanstd(pred)/np.nanstd(obs))/beta
    kge=1-((r-1)**2+(beta-1)**2+(gamma-1)**2)**0.5
    return kge
#%%
def split_sequence_multi_train(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[k:end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1

    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
def split_sequence_multi_s(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[end_ix:out_end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1
    
    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
seq_length = 365
forecast_horizon=7
batch_size = 1024


columns=df_train_norm.columns.to_list()
columns.remove('basin_id')

columns_era=df_train_era_norm.columns.to_list()
columns_era.remove('basin_id')

import random
seed=213
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For CUDA
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random module
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
class HLS(nn.Module):
    def __init__(self):
        """Initializes the instance attributes"""
        super(HLS, self).__init__()
        
        # Initialize S and sigma matrices
        s_temp = torch.diag(torch.ones(7).to(device))
        s_temp = torch.cat((torch.ones(1, 7).to(device), s_temp), dim=0)
        self.S = torch.tensor(s_temp, dtype=torch.float32)

        sigma_temp = torch.diag(torch.ones(8).to(device))
        sigma_temp[0] = sigma_temp[0] * 7
        self.sigma = torch.tensor(sigma_temp, dtype=torch.float32)
        
        # Precompute the inverse of sigma
        self.sigma_inverse = torch.linalg.inv(self.sigma)
        
        # Precompute (Sᵀ * sigma⁻¹ * S)⁻¹
        S_t = self.S.T
        self.S_t_S_inv = torch.linalg.inv(S_t @ self.sigma_inverse @ self.S)

    def forward(self, inputs):
        """
        Defines the computation from inputs to outputs
        :param inputs: torch.Tensor
        :return: torch.Tensor
        """
        # Compute beta = (Sᵀ * sigma⁻¹ * S)⁻¹ * Sᵀ * sigma⁻¹ * inputs.T
        beta = self.S_t_S_inv @ self.S.T @ self.sigma_inverse @ inputs.T
        
        # Reconcile the outputs: reconcile = S * beta
        reconcile = self.S @ beta
        
        # Transpose the reconciled output
        reconcile = reconcile.T
        return reconcile
#%%
class LSTMModelW(nn.Module):
    """
    LSTM model with two inputs processed by separate LSTM layers, 
    with one output (1w) from the encoded spaces.
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, num_layers=1,dropout_prob = 0.4):
        """
        Args:
            input_dim1 (int): Number of features for the first input.
            input_dim2 (int): Number of features for the second input.
            hidden_dim (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModelW, self).__init__()
        
        # LSTM layers for the two inputs
        self.lstm1 = nn.LSTM(input_dim1, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim, num_layers, batch_first=True)
        # Dropout layer for avoiding overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers for 1D and 7D outputs
        self.fc_1d = nn.Linear(hidden_dim+hidden_dim, 1)  # For 7D output

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
        lstm_out_daymet, _ = self.lstm1(input1)  # lstm_out_daymet: [batch_size, seq_len, hidden_dim]
        lstm_out_era5, _ = self.lstm2(input2)    # lstm_out_era5: [batch_size, seq_len, hidden_dim]
        
        # Extract the last hidden states (output at the final time step)
        last_out_daymet = lstm_out_daymet[:, -1, :]  # [batch_size, hidden_dim]
        last_out_era5 = lstm_out_era5[:, -1, :]      # [batch_size, hidden_dim]
        
        # Concatenate the last hidden states along the feature dimension
        combined_features = torch.cat((last_out_daymet, last_out_era5), dim=1)  # [batch_size, hidden_dim * 2]
        
        # Apply dropout
        combined_features = self.dropout(combined_features)
        # Pass through separate linear layers for 1D and 7D outputs
        output_1d = self.fc_1d(combined_features)  # [batch_size, 1]
        #output_7d = self.fc_7d(combined_features)  # [batch_size, 7]
        return output_1d
    
class LSTMModelD(nn.Module):
    """
    LSTM model with two inputs processed by separate LSTM layers, 
    with multiple outputs (7Ds) from the encoded spaces.
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, num_layers=1,dropout_prob = 0.4):
        """
        Args:
            input_dim1 (int): Number of features for the first input.
            input_dim2 (int): Number of features for the second input.
            hidden_dim (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModelD, self).__init__()
        
        # LSTM layers for the two inputs
        self.lstm1 = nn.LSTM(input_dim1, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim, num_layers, batch_first=True)
        # Dropout layer for avoiding overfitting
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers for 1D and 7D outputs
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
        lstm_out_daymet, _ = self.lstm1(input1)  # lstm_out_daymet: [batch_size, seq_len, hidden_dim]
        lstm_out_era5, _ = self.lstm2(input2)    # lstm_out_era5: [batch_size, seq_len, hidden_dim]
        
        # Extract the last hidden states (output at the final time step)
        last_out_daymet = lstm_out_daymet[:, -1, :]  # [batch_size, hidden_dim]
        last_out_era5 = lstm_out_era5[:, -1, :]      # [batch_size, hidden_dim]
        
        # Concatenate the last hidden states along the feature dimension
        combined_features = torch.cat((last_out_daymet, last_out_era5), dim=1)  # [batch_size, hidden_dim * 2]
        
        # Apply dropout
        combined_features = self.dropout(combined_features)
        # Pass through separate linear layers for 1D and 7D outputs
        #output_1d = self.fc_1d(combined_features)  # [batch_size, 1]
        output_7d = self.fc_7d(combined_features)  # [batch_size, 7]
        return output_7d
    
    
class LSTMHLS(nn.Module):
    def __init__(self, pretrained_model_1d, pretrained_model_7d):
        """
        Args:
            pretrained_model_1d (LSTMModel): Pretrained LSTM model for 1D output.
            pretrained_model_7d (LSTMModel): Pretrained LSTM model for 7D output.
            hidden_dim (int): Hidden size of the final HLS layer.
        """
        super(LSTMHLS, self).__init__()

        # Load the two pretrained LSTM models
        self.lstm_1d = pretrained_model_1d
        self.lstm_7d = pretrained_model_7d
        
        # Optionally freeze the pretrained weights (Uncomment if needed)
        for param in self.lstm_1d.parameters():
            param.requires_grad = False
        for param in self.lstm_7d.parameters():
            param.requires_grad = False
        
        # HLS Layer (Processes concatenated outputs)
        self.hls_layer = HLS()  # Combines both outputs


    def forward(self, input1, input2):
        """
        Forward pass through the combined model.
        Args:
            input1 (torch.Tensor): First input tensor [batch_size, seq_length, features].
            input2 (torch.Tensor): Second input tensor [batch_size, seq_length, features].
        
        Returns:
            torch.Tensor: Final output.
        """
        # Get outputs from the two pretrained models
        output_1d = self.lstm_1d(input1, input2)  # [batch_size, 1]
        output_7d = self.lstm_7d(input1, input2)  # [batch_size, 7]
        
        # Concatenate both outputs
        combined_out = torch.cat((output_1d, output_7d), dim=1)  # [batch_size, 8]

        # Pass through HLS Layer
        hls_out = self.hls_layer(combined_out)  # [batch_size, hidden_dim]

        
        return hls_out
#%%


# Model parameters
MODEL_PARAMS = {
    "input_size": 33,  # Dynamic + static + q (past)
    "input_size_era": 64, # All ERA5-Land data
    "hidden_size": 128,
    "num_layers": 1,
    "dropout_prob": 0.4,
}



def load_model(model_class, model_path: Path, **model_params):
    """
    Loads a PyTorch model from a given path.

    Args:
        model_class (torch.nn.Module): The model class to instantiate.
        model_path (Path): Path to the saved model file.
        **model_params: Model parameters to initialize the model.

    Returns:
        torch.nn.Module: Loaded model on the appropriate device.
    """
    model = model_class(**model_params)
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


# Load models
#Weekly model
model_w_path = Path(f"ModelLSTM_Multi_Regional_{seed}_W")
#Daily model
model_d_path = Path(f"ModelLSTM_Multi_Regional_{seed}_BU")

modeliw = load_model(LSTMModelW, model_w_path, **MODEL_PARAMS)
modelid = load_model(LSTMModelD, model_d_path, **MODEL_PARAMS)

# Combine models
modeli_hls = LSTMHLS(modeliw, modelid).to(device)

# Define output directory
output_folder = Path(f"Multiscale_Regional_HLS_{seed}")
output_folder.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist

print(f"Models loaded successfully on {device}. Output directory: {output_folder}")
#%%
def preprocess_test_data(df_test_tr, df_test_tr_era, basin_id, columns, columns_era):
    """
    Extracts and processes test data for a specific basin.

    Args:
        df_test_tr (pd.DataFrame): Transformed test data (main dataset).
        df_test_tr_era (pd.DataFrame): Transformed test data (ERA dataset).
        basin_id (int): The ID of the basin to process.
        columns (list): Columns to extract from df_test_tr.
        columns_era (list): Columns to extract from df_test_tr_era.

    Returns:
        Tuple[torch.Tensor]: Processed test inputs (X, X_era) and target (Y).
    """
    temp_xx = df_test_tr[df_test_tr['basin_id'] == basin_id].loc[:, columns].to_numpy()
    temp_xx_era = df_test_tr_era[df_test_tr_era['basin_id'] == basin_id].loc[:, columns_era].to_numpy()
    temp_yy = df_test_tr[df_test_tr['basin_id'] == basin_id]['q'].to_numpy().reshape((-1, 1))

    xx_, yy_ = split_sequence_multi_train(temp_xx, temp_yy, 365, 7, mode='seq')
    xx_era, _ = split_sequence_multi_s(temp_xx_era, temp_yy, 365, 7, mode='seq')

    return (
        torch.tensor(xx_, dtype=torch.float32).to(device),
        torch.tensor(xx_era, dtype=torch.float32).to(device),
        torch.tensor(yy_, dtype=torch.float32).to(device),
    )


def evaluate_model(model, X_test, X_test_era):
    """
    Performs model inference on the given test data.

    Args:
        model (torch.nn.Module): The trained model.
        X_test (torch.Tensor): Input test data (main dataset).
        X_test_era (torch.Tensor): Input test data (ERA dataset).

    Returns:
        np.ndarray: Model predictions.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test, X_test_era)
    return y_pred.cpu().numpy()


def denormalize_predictions(y_pred, std_q, mean_q):
    """
    Denormalizes the model predictions.

    Args:
        y_pred (np.ndarray): Normalized predictions.
        std_q (float): Standard deviation of the target variable.
        mean_q (float): Mean of the target variable.

    Returns:
        np.ndarray: Denormalized predictions.
    """
    return y_pred * std_q + mean_q


def save_results(output_folder, basin_id, pred_test, pred_test_w, y_test, y_test_w):
    """
    Saves model predictions and ground truth values as a CSV file.

    Args:
        output_folder (Path): The output directory.
        basin_id (int): Basin ID.
        pred_test (np.ndarray): Daily model predictions.
        pred_test_w (np.ndarray): Weekly aggregated predictions.
        y_test (np.ndarray): Daily true values.
        y_test_w (np.ndarray): Weekly aggregated true values.
    """
    df_output = pd.concat([
        pd.DataFrame(pred_test, columns=[f"pr_{i+1}" for i in range(pred_test.shape[1])]),
        pd.DataFrame(pred_test_w, columns=["pr_w"]),
        pd.DataFrame(y_test, columns=[f"obs_{i+1}" for i in range(y_test.shape[1])]),
        pd.DataFrame(y_test_w, columns=["obs_w"])
    ], axis=1)

    output_file = output_folder / f"basin_{basin_id}_results.csv"
    df_output.to_csv(output_file, index=False)


# Main Processing Loop
output_folder = Path(output_folder)  # Ensure it's a Path object

for basin_id in range(421):
    # Load test data for the given basin
    X_test, X_test_era, Y_test = preprocess_test_data(df_test_norm, df_test_era_norm, basin_id, columns, columns_era)

    # Get model predictions
    y_pred = evaluate_model(modeli_hls, X_test, X_test_era)

    # Process predictions
    y_pred_sum = y_pred[:, 0].reshape((-1, 1))

    y_test = denormalize_predictions(Y_test.cpu().numpy(), std_q, mean_q)
    y_test_w = np.sum(y_test, axis=1).reshape((-1, 1))

    pred_test = denormalize_predictions(y_pred[:, 1:], std_q, mean_q)
    pred_test_w = y_pred_sum / 7 * std_q + mean_q
    pred_test_w *= 7  # Convert back to weekly sum

    # Save results
    save_results(output_folder, basin_id, pred_test, pred_test_w, y_test, y_test_w)

print(f"Processing complete. Results saved in {output_folder}")

