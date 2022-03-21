import math

import torch
import torch.nn as nn

class Dense(nn.Module):
    def __init__(
        self,
        input_dim,
        # hidden_dim,
        # kernel_size,
        # padding,
        # pooling,
        # dropout,
        output_class_num,
        **kwargs,
    ):
        super(Dense, self).__init__()
        self.basic_model = nn.Sequential(
            # Dense layer 1
            nn.Dropout(p=0.2),
            nn.Conv1d(input_dim, 128, kernel_size=1, stride=1, padding=0),            
            nn.ReLU(),
            
            # Dense layer 2
            nn.Dropout(p=0.2),
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            # how to avg all the frames in this step?????
            # TODO

            # Output layer: Softmax
            # TODO
            nn.functional.softmax(output_class_num, dim=-1),            
        )

    def forward(self, features, features_len):        
        predicted = self.basic_model(out)
        return predicted


# TODO: Implement "LSTM" type model also, almost identical to "Dense"

"""
Implementation of "Fusion" downstream model presented in the paper
-> Results reported in the paper utilized both global and speaker normalization     
Original Paper: Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings
https://arxiv.org/pdf/????????????????.pdf     2104.03502
"""
class Fusion(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super(Fusion, self).__init__()
        #####TODO

    def forward(self, features, att_mask):
        #TODO
        pass