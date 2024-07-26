import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from sklearn. import LabelEncoder, OneHotEncoder
# import datapipeline


df = pd.read_csv('data/datasets/weatherAUS.csv')
print(df.head())
print(df.info())
# TODO: do smth with obj columns
# TODO: do smth about categorical data
# TODO: Map last cols to 0-1
# df = df.map()


class ReadDataset(Dataset):
    def __init__(self, csv_file):
        # Read data
        self.df = pd.read_csv(csv_file)
        # Transform columns to numerical and stuff
        self.data = self.df.to_numpy()
        # Set X and y, make tensors out of them
        self.X = torch.from_numpy(self.data[:, :-1])
        self.y = torch.from_numpy(self.data[:, -1])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


file = 'data/datasets/weatherAUS.csv'

my_dataset = ReadDataset(file)

dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
for x in enumerate(dataloader):
    print(x)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
class MyDataset(Dataset):
    def __init__(self, root, n_inp):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        self.x , self.y = (torch.from_numpy(self.data[:,:n_inp]),
                           torch.from_numpy(self.data[:,n_inp:]))
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx,:]
    def __len__(self):
        return len(self.data)

myData = MyDataset("data.csv", 20)

data_loader = DataLoader(myData, batch_size=4, shuffle =True)

for x,y in data_loader:
   . . .

"""