from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from src.sentiment_model import SentimentModel
import torch.nn as nn



model = SentimentModel(vocab_size=10000,embedding_dim=100)

#The Judge
criterion = nn.CrossEntropyLoss()


#The Coach
optimizer = optim.Adam(model.parameters,lr=0.001)


# with open("./../train_dataset",'r',encoding='utf-8') as f:
#     train_dataset = 