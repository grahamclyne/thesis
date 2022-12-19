import numpy as np
import torch as T
import torch.nn as nn
from sklearn import preprocessing
import time
from pickle import dump
from sklearn.metrics import mean_squared_error, r2_score
from torchmetrics.functional import r2_score as R2Score
import pandas as pd
from model import CMIPDataset,Net
import other.constants as constants
from other.config import DATA_PATH
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from other.constants import MODEL_TARGET_VARIABLES,MODEL_INPUT_VARIABLES

# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() 
# and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement


def run(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        train, validation = load_data(config)
        model = Net(len(MODEL_INPUT_VARIABLES),len(MODEL_TARGET_VARIABLES))
        loss_function = nn.MSELoss()
        optimizer = T.optim.Adam(model.parameters(), lr=config.lr)
        for epoch in range(config.epochs):
            model.train()
            avg_train_loss = 0
            avg_validation_loss = 0
            for X,y in tqdm(train):
                optimizer.zero_grad() #clears old gradients from previous steps 
                pred_y = model(X.float())
                train_loss = loss_function(pred_y, y.float())
                wandb.log({"training loss": train_loss})
                train_loss.backward() #compute gradient
                optimizer.step() #take step based on gradient
                avg_train_loss += train_loss.item() / X.size(0)
            model.eval()
            for X,y in tqdm(validation):
                pred_y = model(X.float())
                val_loss = loss_function(pred_y,y.float())
                wandb.log({"validation loss": val_loss})
                avg_validation_loss += val_loss.item() / X.size(0)
            wandb.log({
                'epoch': epoch,
                'avg_train_loss':avg_train_loss,
                'avg_validation_loss':avg_validation_loss
            })


def load_data(config):
    #prepare and scale data
    data = pd.read_csv(f'{DATA_PATH}/cesm_data_variant.csv')
    data = data.reset_index()
    data.columns = ['year','lat','lon','cOther','cCwd','cVeg','cLitter','cLeaf','cRoot','evspsblsoi','lai','tsl','mrro','mrsos','grassFrac','shrubFrac','cropFrac','baresoilFrac','residualFrac','treeFrac','shrubFrac','cSoil','cStem','wetlandFrac','ps','pr','tas_DJF','tas_JJA','tas_MAM','tas_SON','grassCropFrac','variant']

    data = data.rename(columns={'# year':'year'})
    ds = data[data['year'] < 2012]
    final_test = data[data['year'] >= 2012] 

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_scaler = preprocessing.StandardScaler()

    input_variables = constants.MODEL_INPUT_VARIABLES
    input_variable_tuple = tuple(constants.MODEL_INPUT_VARIABLES)
    target_variables = constants.MODEL_TARGET_VARIABLES
    #split these so no data is leaked between test and training
    scaler = min_max_scaler.fit(ds.loc[:,input_variables])
    ds.loc[:,input_variable_tuple] = scaler.transform(ds.loc[:,input_variable_tuple])

    final_test.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(final_test.loc[:,input_variable_tuple])

    ds = CMIPDataset(ds[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))
    final_test = CMIPDataset(final_test[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))

    #split train and validation
    train_set_size = int(len(ds) * 0.75)
    valid_set_size = len(ds) - train_set_size
    train,validation = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))
    train_ldr = T.utils.data.DataLoader(train,batch_size=config.batch_size,shuffle=True)
    validation_ldr = T.utils.data.DataLoader(validation,batch_size=config.batch_size,shuffle=True)
    return train_ldr,validation_ldr




if __name__ == "__main__":
   
    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'validation_loss'
		},
    'parameters': {
        'batch_size': {'values': [32,64,128,256]},
        'epochs': {'values': [50, 100, 150,200]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}   
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="ann-sweep")
    wandb.agent(sweep_id, function=run, count=4)



