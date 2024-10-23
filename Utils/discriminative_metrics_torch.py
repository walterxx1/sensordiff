import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def train_test_divide(ori_data, generated_data, ori_time, generated_time):
    """Divide data into train and test sets.
    Args:
        - ori_data: original data
        - generated_data: generated data
        - ori_time: original time information
        - generated_time: generated time information
    Returns:
        - train and test sets for both original and generated data
    """
    train_x, test_x, train_t, test_t = train_test_split(ori_data, ori_time, test_size=0.2)
    train_x_hat, test_x_hat, train_t_hat, test_t_hat = train_test_split(generated_data, generated_time, test_size=0.2)
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def discriminative_score_metrics(ori_data, generated_data, device='cpu'):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - device: 'cpu' or 'cuda' for GPU execution
    
    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
    
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128
    
    # Build a post-hoc RNN discriminator network
    # Define the model
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.dense = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x, t):
            # Pack the sequences to handle variable length sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(x, t.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.gru(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            # Use the last hidden state
            d_last_states = output[range(len(output)), t - 1, :]
            y_hat_logit = self.dense(d_last_states)
            y_hat = self.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat
    
    discriminator = Discriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(discriminator.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    # Convert data to torch tensors and move to device
    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)
    ori_time = torch.tensor(ori_time, dtype=torch.int64).to(device)
    generated_time = torch.tensor(generated_time, dtype=torch.int64).to(device)
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        
        X_mb = X_mb.clone().detach().to(device)
        T_mb = T_mb.clone().detach().to(device)
        X_hat_mb = X_hat_mb.clone().detach().to(device)
        T_hat_mb = T_hat_mb.clone().detach().to(device)
        
        # Train discriminator
        optimizer.zero_grad()
        y_logit_real, _ = discriminator(X_mb, T_mb)
        y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)
        
        # Loss for the discriminator
        real_labels = torch.ones_like(y_logit_real).to(device)
        fake_labels = torch.zeros_like(y_logit_fake).to(device)
        d_loss_real = criterion(y_logit_real, real_labels)
        d_loss_fake = criterion(y_logit_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # Backward and optimize
        d_loss.backward()
        optimizer.step()
    
    # Test the performance on the testing set
    test_x = test_x.clone().detach().requires_grad_(False).to(device)
    test_t = test_t.clone().detach().requires_grad_(False).to(device)
    test_x_hat = test_x_hat.clone().detach().requires_grad_(False).to(device)
    test_t_hat = test_t_hat.clone().detach().requires_grad_(False).to(device)
    
    y_pred_real_curr, _ = discriminator(test_x, test_t)
    y_pred_fake_curr, _ = discriminator(test_x_hat, test_t_hat)
    
    y_pred_final = torch.cat((y_pred_real_curr, y_pred_fake_curr), dim=0).squeeze().detach().cpu().numpy()
    y_label_final = np.concatenate((np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr))), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score
