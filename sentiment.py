import torch
from torch import nn, optim

def test_func():
  print("test")

def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    """ 
    Build a dataloader.
    """
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor



class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(self.embed_size, self.lstm_size, self.lstm_layers, dropout=self.dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """
        
        # TODO Implement 
        
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.lstm_layers, batch_size,self.lstm_size).zero_(),
                         weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        return hidden



    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """
        
        # TODO Implement 
        batch_size = nn_input.size(0)
        
        embeds = self.embedding(nn_input)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        
        #lstm_out = lstm_out.contiguous().view(-1, self.lstm_size)    
        """
        remember here you do not have batch_first=True, 
        so accordingly shape your input. 
        Moreover, since now input is seq_length x batch you just need to transform lstm_out = lstm_out[-1,:,:].
        you don't have to use batch_first=True in this case, 
        nor reshape the outputs with .view just transform your lstm_out as advised and you should be good to go.
        """
        lstm_out = lstm_out[-1,:,:]
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        logps = self.softmax(out)
        
        
        return logps, hidden_state
