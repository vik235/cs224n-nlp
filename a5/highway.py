#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    """ Init the Highway model.
        This part contains the layers of the model(layers that have weights)

        @param word_embed_size (int): Embedding size (dimensionality) of word
        @param dropout_rate (float, between 0 and 1): Dropout probability
    """
    def __init__(self, word_emb_size):
        super(Highway, self).__init__()
        self.input_size = word_emb_size
        #self.dropout_rate = dropout_rate
        self.proj = nn.Linear(self.input_size, self.input_size, bias=True)
        self.gate = nn.Linear(self.input_size, self.input_size, bias=True)
    
    """ Take a mini-batch of the words and run it through the highway layer including the dropout. 

        @param x_conv_out: Output of the convolution layer as referred in the handout. 
        The shape of x_conv_out is (minibatch_size, word_emb_size)
        @returns x_word_embed (Tensor): Result of the forward pass of the highway layer
        of size (minibatch_size, word_emb_size).
    """
    def forward(self, x_conv_out): # x_conv_out = (batch_size, word_emb_size)
        x_proj = self.proj(x_conv_out)
        x_proj = F.relu(x_proj)
        x_gate = self.gate(x_conv_out)
        x_gate = torch.sigmoid(x_gate)
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((1 - x_gate), x_conv_out)
        #x_word_embed = F.dropout(x_highway, self.dropout_rate)
        return x_highway#x_word_embed
        
    ### END YOUR CODE

