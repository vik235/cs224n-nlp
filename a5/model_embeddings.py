#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.num_embed = len(self.vocab.char2id)
        self.char_embedding = 50
        self.padding_idx = vocab.char_pad
        self.embedding = nn.Embedding(num_embeddings = self.num_embed, embedding_dim = self.char_embedding, padding_idx= self.padding_idx)
        self.cnn = CNN(self.char_embedding, num_filters=self.word_embed_size)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
            
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        """
        x_word_embed_list = []
        for x_padded in input:
            x_emb = self.embedding(x_padded)   #(batch_size, max_word_length, char_emb_dim)
            x_reshaped = torch.transpose(x_emb, dim0 = -1 , dim1 = -2)
            x_conv_out = self.cnn(x_reshaped)
            x_highway = self.highway(x_conv_out)
            x_word_embed = self.dropout(x_highway)
            x_word_embed_list.append(x_word_embed)
        stacked_word_emd = torch.stack(x_word_embed_list)
        return stacked_word_emd
        """
         ### YOUR CODE HERE for part 1j
        #print(self.embed_size)
        #print(input.size())
        sentence_length = input.size()[0]
        batch_size = input.size()[1]
        e = self.embedding(input)
        #print(e.size())
        e = e.permute(0, 1, 3, 2)
        e = e.contiguous()
        e = e.view(-1, e.size()[2], e.size()[3])
        #print(e.size())
        x_conv_out = self.cnn.forward(e)
        #print(x_conv_out.size())
        x_highway = self.highway.forward(x_conv_out)
        #print(x_highway.size())
        x_word_emb = self.dropout(x_highway)
        #print(x_word_emb.size())
        x_word_emb = x_word_emb.view(sentence_length, batch_size, self.word_embed_size)
        return x_word_emb
        ### END YOUR CODE

