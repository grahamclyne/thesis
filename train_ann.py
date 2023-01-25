import torch as T
import torch.nn as nn
from sklearn import preprocessing
import pandas as pd
from model import CMIPDataset,Net
from tqdm import tqdm
import wandb
import hydra
from pickle import dump
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="ann_config")
def main(cfg: DictConfig):
    wandb.init(project="land-carbon-ann", config=cfg)
    train, validation = load_data(cfg)
    model = Net(len(cfg.model.input),len(cfg.model.output))
    loss_function = nn.MSELoss()
    optimizer = T.optim.Adam(model.parameters(), lr=cfg.params.lr)
    for epoch in range(cfg.params.epochs):
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
    T.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'ann_checkpoint.pt')
    


def load_data(cfg:DictConfig):
    #prepare and scale data
    data = pd.read_csv(f'{cfg.path.data}/cesm_data_variant.csv')
    data = data.reset_index()
    data.columns = ['year','lat','lon','cOther','cCwd','cVeg','cLitter','cLeaf','cRoot','evspsblsoi','lai','tsl','mrro','mrsos','grassFrac','shrubFrac','cropFrac','baresoilFrac','residualFrac','treeFrac','shrubFrac','cSoil','cStem','wetlandFrac','ps','pr','tas_DJF','tas_JJA','tas_MAM','tas_SON','grassCropFrac','variant']

    data = data.rename(columns={'# year':'year'})
    ds = data[data['year'] < 2012]
    final_test = data[data['year'] >= 2012] 

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_scaler = preprocessing.StandardScaler()

    input_variables = cfg.model.input
    input_variable_tuple = tuple(cfg.model.input)
    target_variables = cfg.model.output
    #split these so no data is leaked between test and training
    scaler = min_max_scaler.fit(ds.loc[:,input_variables])
    ds.loc[:,input_variable_tuple] = scaler.transform(ds.loc[:,input_variable_tuple])

    final_test.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(final_test.loc[:,input_variable_tuple])

    ds = CMIPDataset(ds[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))
    final_test = CMIPDataset(final_test[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))
    dump(scaler, open(f'ann_scaler.pkl','wb'))

    #split train and validation
    train_set_size = int(len(ds) * 0.75)
    valid_set_size = len(ds) - train_set_size
    train,validation = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))
    train_ldr = T.utils.data.DataLoader(train,batch_size=cfg.params.batch_size,shuffle=True)
    validation_ldr = T.utils.data.DataLoader(validation,batch_size=cfg.params.batch_size,shuffle=True)
    return train_ldr,validation_ldr





   
#     sweep_configuration = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'validation_loss'
# 		},
#     'parameters': {
#         'batch_size': {'values': [32,64,128,256]},
#         'epochs': {'values': [50, 100, 150,200]},
#         'lr': {'max': 0.1, 'min': 0.0001}
#      }
# }   
    
#     sweep_id = wandb.sweep(sweep=sweep_configuration, project="ann-sweep")
#     wandb.agent(sweep_id, function=run, count=4)

main()

