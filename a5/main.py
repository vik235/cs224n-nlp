from highway import *
from cnn import *

char_embed_size = 50
word_emb_size = 5
dropout_rate = 0.5
max_word_length = 21
kernel = 5
batch_size=10

input = torch.randn((1,word_emb_size))
batch_input = torch.randn((10,word_emb_size))
print(batch_input)
highway = Highway(word_emb_size, dropout_rate)
print(highway)
out = highway(batch_input)
print(out)
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)



t = torch.randn((1, 5))
print(t[:])
print(t[:-1])