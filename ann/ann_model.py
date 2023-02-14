import torch as T
class CMIPDataset(T.utils.data.Dataset):
    def __init__(self, data,num_of_inputs,num_of_targets):
        self.data = data
        self.num_of_inputs = num_of_inputs
        self.num_of_targets = num_of_targets
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:self.num_of_inputs]
        y = self.data[idx][-self.num_of_targets:]
        return X,y


class Net(T.nn.Module):
  def __init__(self,num_of_inputs,num_of_targets):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(num_of_inputs,30) 
    self.hid2 = T.nn.Linear(30,100)
    # self.hid3 = T.nn.Linear(100,200)
    # self.hid4 = T.nn.Linear(200,100)
    self.batchnorm1 = T.nn.BatchNorm1d(100)

    self.hid5 = T.nn.Linear(100,50)
    self.hid6 = T.nn.Linear(50,25)
    self.hid7 = T.nn.Linear(25,15)
    self.drop1 = T.nn.Dropout(.4) #example of dropout layer

    self.oupt = T.nn.Linear(15, num_of_targets)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))

    # z = T.relu(self.hid3(z))
    # z = T.relu(self.hid4(z))
    z = self.batchnorm1(z)

    z = T.relu(self.hid5(z))
    z = T.relu(self.hid6(z))
    z = T.relu(self.hid7(z))

    z = self.drop1(z)
    z = self.oupt(z)  # no activation bc of regression
    return z
