from locale import YESEXPR
import xarray as xr
import csv
import geopandas 
from shapely.geometry import mapping 
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms

class CMIPDataset(T.utils.data.Dataset):
    def __init__(self, netcdf_paths, shp_file_path, root_dir, transform=None):
        self.data = combine_netcdfs(netcdf_paths,shp_file_path=shp_file_path,root=root_dir)
        self.root_dir = root_dir
        # self.transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)])
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:-1]
        y = self.data[idx][-1]
        # if self.transform is not None:
        #     X = self.transform(X)
        #     y = self.transform(y)
        return X,y


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(5,10)  # 8-(10-10)-1
    self.hid2 = T.nn.Linear(10, 10)
    self.drop1 = T.nn.Dropout(0.50) #example of dropout layer

    self.oupt = T.nn.Linear(10, 1)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = self.drop1(z)
    z = self.oupt(z)  # no activation bc of regression
    return z


def combine_netcdfs(file_paths,shp_file_path,root):
    shp_file = geopandas.read_file(shp_file_path, crs="epsg:4326")
    out = np.array([])
    coords = True
    for file in file_paths:
        ds = xr.open_dataset(root + file)
        var = file.split('_')[0]
        arr = netcdf_to_numpy(ds,var,shp_file,coords)
        if(len(out) == 0):
            out = arr
        else:
            out = np.concatenate((out,arr),1)
        if(coords):
            coords = False
    return out 

def netcdf_to_numpy(netcdf_file,variable,shape_file,needCoords):
    #need to check why mod makes this break ------ long180 = (long360 + 180) % 360 - 180
    netcdf_file['lon'] = netcdf_file['lon'] - 360 if np.any(netcdf_file['lon'] > 180) else netcdf_file['lon']
    netcdf_file = netcdf_file[variable]
    netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    netcdf_file.rio.write_crs("epsg:4326", inplace=True)
    clipped = netcdf_file.rio.clip(shape_file.geometry.apply(mapping), shape_file.crs,drop=True)
    df = clipped.to_dataframe().reset_index()
    df = df[df[variable].notna()]
    if(needCoords):
        array = df[[variable,'lat','lon']].values
    else:
        array = df[[variable]].values
    return array



# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement
def main():
    shape_file = geopandas.read_file('/Users/gclyne/boreal_reduced.shp', crs="epsg:4326")
    # cLeaf = netcdf_to_numpy(xr.open_dataset('data/cLeaf_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'),'cLeaf',shape_file,True)
    # gpp = netcdf_to_numpy(xr.open_dataset('data/gpp_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'),'gpp',shape_file,False)
    # cVeg = netcdf_to_numpy(xr.open_dataset('data/cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'),'cVeg',shape_file,False)
    # rGrowth = netcdf_to_numpy(xr.open_dataset('data/rGrowth_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'),'rGrowth',shape_file,False)
    ds = CMIPDataset([
        'cLeaf_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'gpp_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'rGrowth_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'],
        '/Users/gclyne/boreal_reduced.shp','/Users/gclyne/thesis/data/'
        )

    ds.data = (ds.data - ds.data.mean()) / ds.data.std() #where should this be done? 
    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size
    train,test = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(42))
    # X = np.concatenate((cLeaf,gpp,rGrowth),1)
    # y = cVeg
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = Net()
    loss_function = nn.MSELoss()
    learning_rate = 0.01
    train_ldr = T.utils.data.DataLoader(train,batch_size=10000,shuffle=True)
    test_ldr = T.utils.data.DataLoader(test,batch_size=10000,shuffle=True)
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    # X_train = T.from_numpy(X_train).float()
    # y_train = T.from_numpy(y_train).float()
    # X_test = T.from_numpy(X_test).float()
    # y_test = T.utils.data.Dataloader(T.from_numpy(y_test).float()
    min_valid_loss = np.inf
    for epoch in range(10):
        train_loss = 0
        model.train()
        for X,y in train_ldr:
            pred_y = model(X.float())
            loss = loss_function(pred_y, y.float())
            optimizer.zero_grad() #clears old gradients from previous steps 
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
        losses.append(train_loss)


        model.eval()
        valid_loss = 0
        for X,y in test_ldr:
            pred_y = model(X.float())
            test_loss = loss_function(pred_y,y.float())
            valid_loss += test_loss.item() * X.size(0)

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()


main()