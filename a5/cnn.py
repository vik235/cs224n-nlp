#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g

    """ Init the CNN model.
        This part contains the layers of the model(usually layers that have weights, typically layer API exposed by nn)

        @param char_embed_size (int): embedding size for a letter. 
        @param kernel_size (int): kernel size of the filter of the CNN, defaults to 5
        @param num_filters (int): Output channles after convolving. This per the handpout is supposed to be word_embed_size

    """
    def __init__(self, char_embed_size: int = 50, kernel_size: int = 5, padding: int = 1, num_filters: int = None):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = padding
        self.conv1d = nn.Conv1d(self.char_embed_size, self.num_filters, self.kernel_size, stride=1, padding=self.padding)
        #self.maxpool1d = nn.MaxPool1d(self.max_word_length +2*self.padding - self.kernel_size + 1, padding=self.padding)
    
    """ Take a mini-batch of the words and run it through the CNN layer as shown in the handout. 
        @param x_reshaped: Input tensor received by this layer
        The shape of x_reshaped is (minibatch_size, word_emb_size, max_word_length)
        @returns x_conv_out (Tensor): Result of the forward pass of the CNN layer
        of size (minibatch_size, word_embed_size).
    """
    def forward(self, x_reshaped): #x_reshaped (batch_size, char_embedding(50), max_word_length)
        x_conv = self.conv1d(x_reshaped)
        x_conv = F.relu(x_conv)
        x_conv_out, _ = torch.max(x_conv, dim = 2)
        return torch.squeeze(x_conv_out, -1)
    ### END YOUR CODE

