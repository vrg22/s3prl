'''
Created on 3/26/22 at 6:15 AM
@author: desmondcaulley
@email: dc@tqintelligence.com

Description:
'''

#https://arxiv.org/pdf/2010.13886.pdf ---> go to this link to verify model

import torch
import torch.nn as nn


class SeparableConvDefault1D(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(SeparableConvDefault1D, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


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
        super().__init__()

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
        super().__init__()

        self.num_subblock = num_subblock
        self.depth_point_conv = SeparableConv1D(64, 64, filter_size)
        self.point_conv = nn.Conv1d(n_channels, 64, 1, padding='same')
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        if n_channels != 64:
            self.marblesubblocks = [MarbleNetSubBlock(filter_size, n_channels)]
            self.marblesubblocks.extend([MarbleNetSubBlock(filter_size, 64) for i in range(1, self.num_subblock)])
        else:
            self.marblesubblocks = [MarbleNetSubBlock(filter_size, 64) for i in range(self.num_subblock)]


    def forward(self, x):
        skip_convd = self.batch_norm1(self.point_conv(x))

        for k in range(self.num_subblock):
            x = self.marblesubblocks[k](x)

        x = self.batch_norm2(self.depth_point_conv(x))
        x = x + skip_convd
        x = self.dropout(self.relu_act(x))

        return x

class MarbleNet(nn.Module):
    def __init__(self, num_blocks=3, num_subblocks=2, input_dim=128, num_steps=100, num_classes=3):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_subblocks = num_subblocks
        self.prologue_conv = SeparableConv1D(256, 128, 11)
        self.epilogue_conv = SeparableConv1D(128, 128, num_steps//2)
        self.epilogue2_conv = SeparableConv1D(128, 128, 2)
        #self.final_conv = tfkl.Conv2D(num_classes, (1, 1), padding='valid')
        self.final_conv = nn.Linear(128,num_classes) #Dense is exactly the same as [1x1] conv. num_filters = num_nodes_in_dense
        self.filter_sizes = [13, 15, 17, 19]
        self.marblenet_blocks = [MarbleNetBlock(self.num_blocks, self.filter_sizes[0], n_channels=input_dim)]
        self.marblenet_blocks.extend([MarbleNetBlock(self.num_subblocks, self.filter_sizes[k], n_channels=64) for k in range(1, self.num_blocks)])

        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.relu_act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()

    def forward(self, x, return_logits=True):
        x = self.relu_act(self.batch_norm1(self.prologue_conv(x)))

        for k in range(self.num_blocks):
            x = self.marblenet_blocks[k](x)

        x = self.relu_act(self.batch_norm2(self.epilogue_conv(x)))

        x = self.relu_act(self.batch_norm3(self.epilogue2_conv(x)))

        x = self.final_conv(self.flatten(x))

        if return_logits:
            return x
        else:
            return nn.softmax(x)



if __name__ == "__main__":
    model = MarbleNet(num_blocks=3, num_subblocks=2)
    #input = torch.rand((15,661, 256))
    input = torch.rand((15,256, 661))
    model(input)
    model.build(input_shape=(None, 100, 40))
    model.summary()