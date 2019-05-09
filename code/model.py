import torch
import torch.nn as nn
from torch.nn import functional as F


class TextClassificationModel(nn.Module):
    def __init__(self, 
                 embedding_matrix,
                 hidden_dim,
                 da,
                 r,
                 output_size,
                 dropout,
                 num_layers=1,
                 use_lstm=True, 
                 bidirectional=True,
                 train_embedding=True):
        super(TextClassificationModel, self).__init__()
        """
        A text classification model, made of an embedding matrix, one or several recurrent layers
        and a self-attention layer.
        
        Arguments:
        - embedding_matrix: pre-trained embedding matrix of size (vocab_size, embedding_dim).
        - hidden_dim: an integer giving the dimension of the hidden state of the recurrent layer.
        - da : Number of units in the Attention mechanism.
        - r : Number of Attention heads.
        - output_size: an integer giving the size of the output (2 for binary classification).
        - dropout: a float between 0.0 and 1.0 giving the dropout rate.
        - num_layers: (Optional) number of stacked recurrent layers. Default 1.
        - use_lstm: (Optional) boolean that indicates wether to use a LSTM layer or not. 
          When False, use a GRU instead. Default True.
        - bidirectional: (Optional) boolean for using a bidirectional recurrent layer. Default True.
        - train embedding: (Optional) boolean to know if we fine tune the embedding matrix. 
          Default True.
        """
        embedding_dim = embedding_matrix.shape[1]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_lstm = use_lstm
        self.da = da
        self.r = r

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        if train_embedding:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False

        # Recurrent layer
        if use_lstm:
            self.rnn = nn.LSTM(input_size=embedding_dim, 
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional,
                               dropout=dropout,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=embedding_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              batch_first=True)
            
        # Fully connected layer
        self.fully_connected = nn.Linear(r * hidden_dim * self.num_directions, output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        

    def forward(self, x):
        """  
        Perform a forward pass
        
        Arguments:
        - X: tensor of shape (batch_size, sequence_length)
        
        Returns:
        - Output of the linear layer of shape (batch_size, output_size)
        """
        
        # 1. Embeddings layer + dropout
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = self.dropout_layer(x)  # [batch_size, seq_len, embed_dim]
        
        # 2. Recurrent layer(s)
        # First, initialize hidden and cell states.
        # Note that only the LSTM requires the cell state.
        # x is of shape [batch_size, seq_len, hidden_size]
        h0, c0 = self._init_hidden(self.num_layers, x.shape[0], self.hidden_dim)
        if self.use_lstm:
            x, (fn, cn) = self.rnn(x, (h0, c0))
        else:
            x, fn = self.rnn(x, h0)
            
        # 3. Attention layer + dropout
        x = self.self_attention(x, self.da, self.r)  # [batch_size, r, hidden_dim] 
        x = self.dropout_layer(x)
        
        # 4. Final layer
        output = self.fully_connected(x.view(x.size()[0], -1))  # [batch_size, 2]
        
        return output
    
    
    def self_attention(self, x, da, r):
        """
        Attention mechanism in our model. 
        Attention is used to compute soft alignment scores between each of 
        the hidden_state and the last hidden_state of the LSTM. 

        Arguments:
        - lstm_output : Output of the LSTM of shape (batch, seq_len, num_directions * hidden_size).
          Tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        - da : Number of units in the Attention mechanism.
        - r : Number of Attention heads.
        
        Returns:
        - Tensor of size [batch_size, seq_len, r]
        """
        hidden_dim = x.size()[2]
        W_s1 = nn.Linear(hidden_dim, da)
        W_s2 = nn.Linear(da, r)
        
        weight_matrix = F.tanh(W_s1(x))  # [batch_size, seq_len, da]
        weight_matrix = W_s2(weight_matrix)  # [batch_size, seq_len, r]
        weight_matrix = F.softmax(weight_matrix, dim=1)  # [batch_size, seq_len, r]
        weight_matrix = weight_matrix.permute(0, 2, 1)  # [batch-size, r, seq_len]
        
        x = torch.bmm(weight_matrix, x)  # [batch_size, r, hidden_dim]

        return x
    
    
    def _init_hidden(self, num_layers, batch_size, hidden_dim):
        """
        Initialize hidden states for the recurrent layers
        
        Arguments:
        - num_layers : number of stacked layers (int).
        - batch_size : batch size (int).
        - hidden_dim : hidden dimension (int).
        
        Returns:
        - A tuple (h0, c0) containing hidden and cell states.
        """
        
        # The hidden state is twice as large for bidirectional LSTM
        h0 = torch.zeros(num_layers * self.num_directions, batch_size, hidden_dim)
        c0 = torch.zeros(num_layers * self.num_directions, batch_size, hidden_dim)
            
        return (h0, c0)