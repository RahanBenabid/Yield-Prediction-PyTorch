import sys
sys.path.insert(0, './local_modules/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("./data/yield_final.csv")

# print stuff to make sure
# print(df.shape)
# print(df.head())
# print(df.isnull().sum())

# data preprocessing
# drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'Area', 'Item', 'Year'])
# print(df)

# (optional) compute the mean for each NaN row and fill it
# df.fillna(df.mean(), inplace=True)
# print(df)

# we create a variable X that contains all the colums except the hg/ha_yield which is the target in this case
X = df.drop(columns=['hg/ha_yield']).values
y = df['hg/ha_yield'].values

# split the test and training segment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to pytorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# define LinearRegressionModel
class LinearRegressionModel(nn.Module):
	def __init__(self, input_dim):
		super(LinearRegressionModel, self).__init__()
		self.linear == nn.Linear(in_features=X_train.shape[1], out_features=1)
	
	def forward(self, x):
		return self.linear(x)
	
# create an instance of the model
model = LinearRegressionModel()