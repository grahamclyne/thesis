
from torch import nn, Tensor
import torch
import math
import pandas as pd
import numpy as np
from omegaconf import DictConfig

class CMIPTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data:np.ndarray,seq_len:int,num_features:int,cfg:DictConfig):
        self.data = data.to_numpy().reshape(-1,seq_len,num_features)
        self.cfg = cfg
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if(idx > len(self.data) - 5):
        #     raise StopIteration
        source = self.data[idx,:,:len(self.cfg.model.input)]
        target = self.data[idx,-1,-(len(self.cfg.model.output)+len(self.cfg.model.id)):-len(self.cfg.model.id)]
        id = self.data[idx,-1,-len(self.cfg.model.id):]
        return source,target,id


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [ batch size, seq len, embed dim]
            output: [batch size, seq len, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)




class TimeSeriesTransformer(nn.Module):

    def __init__(self, 
        input_feature_size: int,
        input_seq_len: int,
        batch_first: bool,
        dim_val,  
        n_encoder_layers,
        n_heads,
        dropout_encoder,
        dropout_pos_enc,
        dim_feedforward_encoder,
        num_predicted_features
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder


            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        self.input_seq_len = input_seq_len
        self.encoder_input_layer = nn.Linear(
            in_features=input_feature_size, 
            out_features=dim_val 
            )

        self.positional_encoding_layer = PositionalEncoding(
            d_model=dim_val,dropout=dropout_pos_enc)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers            
        )

        self.linear1 = torch.nn.Linear(dim_val,1)
        self.linear2 = torch.nn.Linear(self.input_seq_len,num_predicted_features)



    def forward(self, src: Tensor, src_mask: Tensor=None) -> Tensor:
        """
        Return [ batch_size,target_sequence_length, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence

        """
        src = self.encoder_input_layer(src) 
        src = self.positional_encoding_layer(src) 
        src = self.encoder(src,src_mask)           
        output = self.linear1(src)
        output = output.reshape(-1,self.input_seq_len)
        output = self.linear2(output)
        return output

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

