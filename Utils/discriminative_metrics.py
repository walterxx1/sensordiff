"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
# from .utils import train_test_divide, extract_time, batch_generator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# def discriminative_score_metrics (ori_data, generated_data):
#   """Use post-hoc RNN to classify original data and synthetic data
  
#   Args:
#     - ori_data: original data
#     - generated_data: generated synthetic data
    
#   Returns:
#     - discriminative_score: np.abs(classification accuracy - 0.5)
#   """
#   # Initialization on the Graph
#   # tf.reset_default_graph()

#   # Basic Parameters
#   no, seq_len, dim = np.asarray(ori_data).shape    
    
#   # Set maximum sequence length and each sequence length
#   ori_time, ori_max_seq_len = extract_time(ori_data)
#   generated_time, generated_max_seq_len = extract_time(ori_data)
#   max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
#   ## Builde a post-hoc RNN discriminator network 
#   # Network parameters
#   hidden_dim = int(dim/2)
#   iterations = 2000
#   batch_size = 128
    
#   # Input place holders
#   # Feature
#   # X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
#   X = tf.Variable(initial_value=tf.zeros([batch_size, max_seq_len, dim]), dtype=tf.float32, name="myinput_x")
#   # X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
#   X_hat = tf.Variable(initial_value=tf.zeros([batch_size, max_seq_len, dim], dtype=tf.float32), name="myinput_x")

  
#   # T = tf.placeholder(tf.int32, [None], name = "myinput_t")
#   T = tf.Variable(initial_value=tf.zeros([batch_size], dtype=tf.int32), dtype=tf.int32, name = "myinput_t")
#   # T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
#   T_hat = tf.Variable(initial_value=tf.zeros([batch_size], dtype=tf.int32), dtype=tf.int32, name = "myinput_t_hat")
    
    
    
#   # discriminator function
#   # def discriminator (x, t):
#   #   """Simple discriminator function.
    
#   #   Args:
#   #     - x: time-series data
#   #     - t: time information
      
#   #   Returns:
#   #     - y_hat_logit: logits of the discriminator output
#   #     - y_hat: discriminator output
#   #     - d_vars: discriminator variables
#   #   """
#   #   with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
#   #     d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
#   #     d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
#   #     y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
#   #     y_hat = tf.nn.sigmoid(y_hat_logit)
#   #     d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
#   #   return y_hat_logit, y_hat, d_vars
  
#   def discriminator(x, t):
#       """Simple discriminator function.
      
#       Args:
#         - x: time-series data (batch_size, seq_len, features)
#         - t: time information (batch_size,)
        
#       Returns:
#         - y_hat_logit: logits of the discriminator output
#         - y_hat: discriminator output
#       """
#       # Use GRU layer instead of GRUCell + dynamic_rnn
#       gru_layer = tf.keras.layers.GRU(units=hidden_dim, activation='tanh', return_sequences=False, return_state=True)
      
#       t = tf.minimum(t, tf.shape(x)[1])
#       # Apply GRU and mask using the time information (sequence lengths)
#       mask = tf.sequence_mask(t, maxlen=tf.shape(x)[1])
#       d_outputs, d_last_states = gru_layer(x, mask=mask)
      
#       # Fully connected layer (equivalent to fully_connected in TensorFlow 1.x)
#       y_hat_logit = tf.keras.layers.Dense(1, activation=None)(d_last_states)
      
#       # Apply sigmoid to get the final output
#       y_hat = tf.nn.sigmoid(y_hat_logit)
      
#       # Variables are automatically tracked in TensorFlow 2.x, so no need for manual variable scope
#       d_vars = gru_layer.trainable_variables# + tf.keras.layers.Dense(1).trainable_variables
      
#       return y_hat_logit, y_hat, d_vars
  
    
#   y_logit_real, y_pred_real, d_vars = discriminator(X, T)
#   y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
#   # Loss for the discriminator
#   d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
#                                                                        labels = tf.ones_like(y_logit_real)))
#   d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
#                                                                        labels = tf.zeros_like(y_logit_fake)))
#   d_loss = d_loss_real + d_loss_fake
    
#   # optimizer
#   # d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
#   optimizer = tf.keras.optimizers.Adam()
#   d_solver = optimizer.minimize(d_loss, var_list=d_vars)
  
#   ## Train the discriminator   
#   # Start session and initialize
#   # sess = tf.Session()
#   # sess.run(tf.global_variables_initializer())
    
#   # Train/test division for both original and generated data
#   train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
#   train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
#   # Training step
#   for itt in range(iterations):
          
#     # Batch setting
#     X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
#     X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
#     # Train discriminator
#     _, step_d_loss = sess.run([d_solver, d_loss], 
#                               feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
    
#   ## Test the performance on the testing set    
#   y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
#                                                 feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
#   y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
#   y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
#   # Compute the accuracy
#   acc = accuracy_score(y_label_final, (y_pred_final>0.5))
#   discriminative_score = np.abs(0.5-acc)
    
#   return discriminative_score  


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
    return np.array(X_mb), np.array(T_mb)


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


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
    
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
    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = tf.keras.layers.GRU(units=hidden_dim, activation='tanh', return_sequences=False)
            self.dense = tf.keras.layers.Dense(1, activation=None)
            
        def call(self, x, t):
            t = tf.cast(tf.minimum(t, tf.cast(tf.shape(x)[1], tf.int64)), tf.int32)  # Ensure t is int32 for compatibility
            mask = tf.sequence_mask(t, maxlen=tf.shape(x)[1])
            d_last_states = self.gru(x, mask=mask)
            y_hat_logit = self.dense(d_last_states)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat
    
    discriminator = Discriminator(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    # Training step
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        
        with tf.GradientTape() as tape:
            # Train discriminator
            y_logit_real, _ = discriminator(X_mb, T_mb)
            y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)
            
            # Loss for the discriminator
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_real, 
                                                                                labels=tf.ones_like(y_logit_real)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_fake, 
                                                                                labels=tf.zeros_like(y_logit_fake)))
            d_loss = d_loss_real + d_loss_fake
        
        # Apply gradients
        gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    # Test the performance on the testing set
    y_pred_real_curr, _ = discriminator(test_x, test_t)
    y_pred_fake_curr, _ = discriminator(test_x_hat, test_t_hat)
    
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]), np.zeros([len(y_pred_fake_curr), ])), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score