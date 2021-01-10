import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

import torch.nn as nn
from collections import OrderedDict

from sklearn.datasets import fetch_20newsgroups

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
        
class Model_20(nn.Module):

    def __init__(self, vocab_size, dim, embeddings):
        super(Model_20, self).__init__()
        
        categories = ['alt.atheism',
         'comp.graphics',
         'comp.os.ms-windows.misc',
         'comp.sys.ibm.pc.hardware',
         'comp.sys.mac.hardware',
         'comp.windows.x',
         'misc.forsale',
         'rec.autos',
         'rec.motorcycles',
         'rec.sport.baseball',
         'rec.sport.hockey',
         'sci.crypt',
         'sci.electronics',
         'sci.med',
         'sci.space',
         'soc.religion.christian',
         'talk.politics.guns',
         'talk.politics.mideast',
         'talk.politics.misc',
         'talk.religion.misc']

        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, 
                                             categories=categories,)

        texts = []

        labels=newsgroups_test.target
        texts = newsgroups_test.data

        MAX_SEQUENCE_LENGTH = 1000
        MAX_NB_WORDS = 20000

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # Create embeddings
        embeddings_index = {}

        path = "/mnt/dsi_vol1/users/frenkel2/data/calibration/20newsgroups/embeddings/"

        with open(path + "glove.6B.100d.txt") as f:
          for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        EMBEDDING_DIM = 100

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        
        self.vocab_size = vocab_size 
        self.dim = dim
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.convnet = nn.Sequential(OrderedDict([
            #('embed1', nn.Embedding(self.vocab_size, self.dim)),
            ('c1', nn.ConvTranspose1d(100, 128, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(128, 128, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(128, 128, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35)),
        ]))
    
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embeddings))
        #copy_((embeddings))
        self.embedding.weight.requires_grad = False
    
        self.fc = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(128, 128)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(128, 20)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        
        output = self.embedding(img)
        output.transpose_(1,2)
        output = self.convnet(output)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        
        return output
