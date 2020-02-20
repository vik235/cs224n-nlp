#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=self.target_vocab.char_pad)


    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        decoderCharEmb = self.decoderCharEmb(input) # -> (length, batch_size, char_embedding_size), x_1..x_n
        hidden_states, dec_hidden = self.charDecoder(decoderCharEmb, dec_hidden) # -> dec_hidden:(1, batch_size, hidden_size)
        scores = self.char_output_projection(hidden_states)# -> (length, batch_size, vocab_size)
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        inpt = char_sequence.narrow(0, 0, char_sequence.size()[0] - 1)
        #print(inpt)
        #print(inpt.size())
        target = char_sequence.narrow(0, 1, char_sequence.size()[0] - 1)
        #print(target)
        #print(target.size())
        scores, dec_hidden = self.forward(inpt, dec_hidden)
        #target = self.decoderCharEmb(target)
        #print(target.size())
        #print(scores.size())
        target = target.contiguous()
        target = target.view(target.size()[0] * target.size()[1])
        scores = scores.view(scores.size()[0] * scores.size()[1], scores.size()[2])
        #print(target.size())
        #print(scores.size())
        loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char_pad)
        return loss(scores, target)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_chars = []
        for i in range(initialStates[0].size()[1]):
            output_chars.append([])
        current_char = [[self.target_vocab.start_of_word] * initialStates[0].size()[1]]
        dec_hidden = initialStates
        for i in range(max_length):
            current_char = torch.tensor(current_char, device=device)
            scores, dec_hidden = self.forward(current_char, dec_hidden=dec_hidden)
            softmax = nn.Softmax(dim=2)
            p = softmax(scores)
            current_char = torch.argmax(p, dim=2)
            chars = current_char.tolist()[0]
            for i in range(len(chars)):
                output_chars[i].append(chars[i])
        output_words = []
        for chars_list in output_chars:
            word = ""
            for c in chars_list:
                if c != self.target_vocab.end_of_word and c!= self.target_vocab.char_pad:
                    word += self.target_vocab.id2char[c]
                else:
                    break
            output_words.append(word)

        return output_words


        
        ### END YOUR CODE

