class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size 
        self.embedding = embedding 
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
    
    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = gru(packed, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, :self.hidden_size]
        return outputs, hidden 

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method 
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        
        def dot_score(self, hidden, encoder_ouput):
            return torch.sum(hidden * encoder_ouput, dim=2)
        
        def general_score(self, hidden, encoder_ouput):
            energy = self.attn(encoder_ouput)
            return torch.sum(hidden * energy, dim=2)
        
        def concat_score(self, hidden, encoder_output):
            energy = self.attn()
        
        


