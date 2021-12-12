import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.input_dims = self.in_seq_len * self.d_model
        self.output_dims = self.out_seq_len * self.d_model
        self.dims_1 = (self.input_dims - self.output_dims) // 4 * 3
        self.dims_2 = (self.input_dims - self.output_dims) // 4 * 2

        linear1 = nn.Linear(self.input_dims, self.dims_1)
        linear2 = nn.Linear(self.dims_1, self.dims_2)
        linear3 = nn.Linear(self.dims_2, self.output_dims)

        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList([linear1, linear2, linear3])
        self.dropout = dropout
        self.unflatten = nn.Unflatten(1, (self.out_seq_len, self.d_model))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
            x = nn.Dropout(p=self.dropout)(x)
        x = self.unflatten(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.input_dims = self.in_seq_len * self.d_model
        self.output_dims =  self.out_seq_len * self.d_model
        self.dims_1 = (self.input_dims - self.output_dims) // 4 * 3
        self.dims_2 = (self.input_dims - self.output_dims) // 4 * 2

        linear1 = nn.Linear(self.output_dims, self.dims_2)
        linear2 = nn.Linear(self.dims_2, self.dims_1)
        linear3 = nn.Linear(self.dims_1, self.input_dims)

        self.linears = nn.ModuleList([linear1, linear2, linear3])
        self.dropout = dropout
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (self.in_seq_len, self.d_model))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
            x = nn.Dropout(p=self.dropout)(x)
        x = self.unflatten(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(self.encoder(input))
        return output


def create_autoencoder(in_seq_len, out_seq_len, d_model, dropout=0.1):
    encoder = Encoder(in_seq_len, out_seq_len, d_model, dropout)
    decoder = Decoder(in_seq_len, out_seq_len, d_model, dropout)
    model = Autoencoder(encoder, decoder)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
