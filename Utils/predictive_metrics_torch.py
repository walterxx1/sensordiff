import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error


def extract_time(data):
    """Extract the time information from the data.
    Args:
        - data: input time-series data
    Returns:
        - time: length of each sequence
        - max_seq_len: maximum sequence length
    """
    time = [len(seq) for seq in data]
    max_seq_len = max(time)
    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Generate a batch of data.
    Args:
        - data: input time-series data
        - time: sequence length information
        - batch_size: number of sequences per batch
    Returns:
        - X_mb: batch data
        - T_mb: batch time information
    """
    idx = np.random.permutation(len(data))[:batch_size]
    X_mb = [data[i] for i in idx]
    T_mb = [time[i] for i in idx]
    return torch.stack(X_mb), torch.tensor(T_mb)


def predictive_score_metrics(ori_data, generated_data, device='cpu'):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - device: 'cpu' or 'cuda' for GPU execution
    
    Returns:
        - predictive_score: MAE of the predictions on the original data
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
    
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128
    
    # Build a post-hoc RNN predictor network
    class Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Predictor, self).__init__()
            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.dense = nn.Linear(hidden_dim, 1)
        
        def forward(self, x, t):
            # Pack the sequences to handle variable length sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(x, t.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.gru(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            # Predict the next step
            y_hat_logit = self.dense(output)
            return y_hat_logit
    
    predictor = Predictor(input_dim=dim - 1, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(predictor.parameters())
    criterion = nn.L1Loss()
    
    # Convert data to torch tensors and move to device
    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)
    ori_time = torch.tensor(ori_time, dtype=torch.int64).to(device)
    generated_time = torch.tensor(generated_time, dtype=torch.int64).to(device)
    
    # Training using synthetic dataset
    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(generated_data, generated_time, batch_size)
        Y_mb = X_mb[:, 1:, -1].unsqueeze(-1)
        X_mb = X_mb[:, :-1, :-1]
        T_mb = T_mb - 1
        
        X_mb = X_mb.clone().detach().to(device)
        T_mb = T_mb.clone().detach().to(device)
        Y_mb = Y_mb.clone().detach().to(device)
        
        # Train predictor
        optimizer.zero_grad()
        y_pred = predictor(X_mb, T_mb)
        p_loss = criterion(y_pred, Y_mb)
        
        # Backward and optimize
        p_loss.backward()
        optimizer.step()
    
    # Test the trained model on the original data
    X_mb, T_mb = batch_generator(ori_data, ori_time, no)
    Y_mb = X_mb[:, 1:, -1].unsqueeze(-1)
    X_mb = X_mb[:, :-1, :-1]
    T_mb = T_mb - 1
    
    X_mb = X_mb.clone().detach().to(device)
    T_mb = T_mb.clone().detach().to(device)
    Y_mb = Y_mb.clone().detach().to(device)
    
    # Prediction
    pred_Y_curr = predictor(X_mb, T_mb)
    
    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), pred_Y_curr[i].cpu().detach().numpy())
    
    predictive_score = MAE_temp / no
    
    return predictive_score
