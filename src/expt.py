import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dvc.api
from sklearn.model_selection import train_test_split

# reading dataset
with dvc.api.open(path = "data/creditcard.csv", repo = "https://github.com/sanidhyaanand/MLOps_Assignment") as fd:
    df = pd.read_csv(fd)

#print(df.columns)

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=42, stratify=Y) 

x = dvc.api.get_url(repo="https://github.com/sanidhyaanand/MLOps_Assignment", path="data/creditcard.csv")
print(x)
