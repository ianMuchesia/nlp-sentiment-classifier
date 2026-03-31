import torch
import torch.nn as nn


class SentimentModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim = 100):
        super(SentimentModel,self).__init__()
        
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(in_features=100,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self,x):
        embedded = self.embedding(x)
        
        #create the mast: (Batch, 250,100)
        #1 for real words, 0 for padding
        mask = ( x!= 0).unsqueeze(-1).float()
        
        
        #apply mask: zero out the padding vectors
        masked_embedded = embedded * mask
        
        
        #sum the vectors and the mask counts
        sum_vectors = torch.sum(masked_embedded,dim=1)
        word_counts = torch.sum(mask,dim=1)
        
        # We add a tiny number (1e-9) to avoid dividing by zero if a review is empty
        x = sum_vectors / (word_counts + 1e-9)
        
        
        #x = torch.mean(x,  dim=1)
        
        x = self.fc1(x)
        
        x = self.Relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        #x = torch.softmax(x)
        
        return x
        
   