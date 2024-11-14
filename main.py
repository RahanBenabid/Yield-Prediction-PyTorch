# import warnings
# warnings.filterwarnings('ignore')
# import seaborn as sns
# import geopandas as gpd

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

# print to make sure
print(df.shape)
print(df.head())

