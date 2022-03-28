'''
Created on 3/26/22 at 6:15 AM
@author: desmondcaulley
@email: dc@tqintelligence.com

Description:
'''

#https://arxiv.org/pdf/2010.13886.pdf ---> go to this link to verify model

import torch
import torch.nn as nn



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = 64
        self.query = nn.Linear(input_dim, self.d_model)
        self.key = nn.Linear(input_dim, self.d_model)
        self.value = nn.Linear(input_dim, self.d_model)
        self.softmax = nn.functional.softmax
        self.multihead_attn = nn.MultiheadAttention(self.d_model, num_heads=1, batch_first=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        key = self.key(x)
        value = self.value(x)
        query = self.query(x)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(1, 2)


        return attn_output

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        batch_rep = batch_rep.transpose(1, 2)
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class SeparableConv1D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        super(SeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(c_in, c_in, kernel_size=kernel_size, padding='same', groups=c_in)
        self.pointwise = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MarbleNetSubBlock(nn.Module):
    def __init__(self, filter_size, n_channels):
        super(MarbleNetSubBlock, self).__init__()

        self.depth_point_conv = SeparableConv1D(n_channels, 64, filter_size)
        self.batch_norm = nn.BatchNorm1d(64)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # needed a squeeze since separableconv2d needed input with max dim of 4
        x = self.batch_norm(self.depth_point_conv(x))
        x = self.dropout(self.relu_act(x))

        return x

class MarbleNetBlock(nn.Module):
    def __init__(self, num_subblock=2, filter_size=13, n_channels=64):
        super(MarbleNetBlock, self).__init__()

        self.num_subblock = num_subblock
        self.depth_point_conv = SeparableConv1D(64, 64, filter_size)
        #self.point_conv = nn.Conv1d(n_channels, 64, 1, padding='same')
        self.self_attn = MultiHeadSelfAttention(n_channels)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        if n_channels != 64:
            self.marblesubblocks = [MarbleNetSubBlock(filter_size, n_channels)]
            self.marblesubblocks.extend([MarbleNetSubBlock(filter_size, 64) for i in range(1, self.num_subblock)])
        else:
            self.marblesubblocks = [MarbleNetSubBlock(filter_size, 64) for i in range(self.num_subblock)]

        self.marblesubblocks = nn.ModuleList(self.marblesubblocks)


    def forward(self, x):
        skip_convd = self.batch_norm1(self.self_attn(x))

        for k in range(self.num_subblock):
            x = self.marblesubblocks[k](x)

        x = self.batch_norm2(self.depth_point_conv(x))
        x = x + skip_convd
        x = self.dropout(self.relu_act(x))

        return x


class MarbleNet(nn.Module):
    def __init__(self, input_dim=256, num_blocks=3, num_subblocks=2, num_steps=100, output_dim=3, pooling=2):
        super(MarbleNet, self).__init__()

        self.num_blocks = num_blocks
        self.num_subblocks = num_subblocks
        self.prologue_conv = SeparableConv1D(input_dim, 128, 11)
        self.epilogue_conv = MultiHeadSelfAttention(64)
        self.epilogue2_conv = AdditiveAttention(64)
        #self.final_conv = tfkl.Conv2D(num_classes, (1, 1), padding='valid')
        self.final_conv = nn.Linear(64,output_dim) #Dense is exactly the same as [1x1] conv. num_filters = num_nodes_in_dense
        self.filter_sizes = [13, 15, 17, 19]
        self.marblenet_blocks = [MarbleNetBlock(self.num_blocks, self.filter_sizes[0], n_channels=128)]
        self.marblenet_blocks.extend([MarbleNetBlock(self.num_subblocks, self.filter_sizes[k], n_channels=64) for k in range(1, self.num_blocks)])
        self.marblenet_blocks = nn.ModuleList(self.marblenet_blocks)

        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, fl, return_logits=True):
        x = x.transpose(1, 2)
        x = self.relu_act(self.batch_norm1(self.prologue_conv(x)))

        for k in range(self.num_blocks):
            x = self.marblenet_blocks[k](x)

        x = self.relu_act(self.batch_norm2(self.epilogue_conv(x)))

        x = self.relu_act(self.batch_norm3(self.epilogue2_conv(x)))

        x = self.final_conv(x)

        return x, None



if __name__ == "__main__":
    model = MarbleNet(num_blocks=3, num_subblocks=2, input_dim=256).to('cuda')
    #input = torch.rand((15,661, 256))
    input = torch.rand((7, 100, 256)).to('cuda')
    model(input)