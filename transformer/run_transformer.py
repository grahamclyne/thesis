import torch as T
import numpy as np 
from tqdm import tqdm
from transformer.transformer_model import TimeSeriesTransformer,generate_square_subsequent_mask,CMIPTimeSeriesDataset
import wandb
from omegaconf import DictConfig
import hydra
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pickle
import pandas as pd
import time



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    

    #init variables
    wandb.init(project="transformer-land-carbon", entity="gclyne",config=cfg)
    T.set_printoptions(sci_mode=False)
    T.set_num_threads(8)
    cesm_df = pd.read_csv(f'{cfg.path.data}/cesm_transformer_data.csv')
    #need to reshape here for data scaling
    # split data into 6 chunks, use 4 for training, 1 for validation, 1 for hold out
    # chunk_size = len((np.array_split(cesm_df, 6, axis=0))[0])
    chunk_size = len(cesm_df)//6
    min_max_scaler = preprocessing.MinMaxScaler((-0.5,0.5))
    hold_out = cesm_df[-chunk_size:]
    train_ds = cesm_df[:chunk_size*5]
    #fix that you are modifying targets here too
    scaler = min_max_scaler.fit(train_ds.loc[:,cfg.model.input])
    hold_out.loc[:,cfg.model.input] = min_max_scaler.fit_transform(hold_out.loc[:,cfg.model.input])
    train_ds.loc[:,cfg.model.input] = min_max_scaler.transform(train_ds.loc[:,cfg.model.input])
    hold_out = hold_out[cfg.model.input + cfg.model.output + cfg.model.id]
    train_ds = train_ds[cfg.model.input + cfg.model.output + cfg.model.id]
    hold_out = CMIPTimeSeriesDataset(hold_out,cfg.trans_params.input_seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    train_ds = CMIPTimeSeriesDataset(train_ds,cfg.trans_params.input_seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    train,validation = T.utils.data.random_split(train_ds, [int((chunk_size/30)*4), int(chunk_size/30)], generator=T.Generator().manual_seed(0))



    model = TimeSeriesTransformer(
        input_feature_size=len(cfg.model.input),
        input_seq_len=cfg.trans_params.input_seq_len,
        batch_first=cfg.trans_params.batch_first,
        dim_val=cfg.trans_params.dim_val,
        n_encoder_layers=cfg.trans_params.num_encoder_layers,
        n_heads=cfg.trans_params.num_heads,
        num_predicted_features=len(cfg.model.output),
        dropout_encoder=cfg.trans_params.dropout_encoder,
        dropout_pos_enc=cfg.trans_params.dropout_pos_enc,
        dim_feedforward_encoder=cfg.trans_params.dim_feedforward_encoder
                   )
    #init weights for each layer in transformer
    for p in model.parameters():
        if p.dim() > 1:
            T.nn.init.xavier_uniform_(p.data)

    #get subset of data for testing
    train = T.utils.data.Subset(train,range(0,cfg.trans_params.batch_size*5))
    validation = T.utils.data.Subset(validation,range(0,cfg.trans_params.batch_size*5))
    hold_out = T.utils.data.Subset(hold_out,range(0,cfg.trans_params.batch_size*5))
    
    
    loss_function = T.nn.MSELoss()
    train_ldr = T.utils.data.DataLoader(train,batch_size=cfg.trans_params.batch_size,shuffle=True)
    validation_ldr = T.utils.data.DataLoader(validation,batch_size=cfg.trans_params.batch_size,shuffle=True)
    hold_out_ldr = T.utils.data.DataLoader(hold_out,batch_size=cfg.trans_params.batch_size,shuffle=True)
    optimizer = T.optim.Adam(model.parameters(), lr=cfg.trans_params.lr)

    src_mask = generate_square_subsequent_mask(
        dim1=cfg.trans_params.input_seq_len,
        dim2=cfg.trans_params.input_seq_len
    )

    #train and validate
    for epoch_count in range(cfg.trans_params.epochs):
        total_start = time.time()

        print('Epoch:',epoch_count)
        train_loss = 0
        model.train()
        for src,tgt,id in train_ldr:
            optimizer.zero_grad() #clears old gradients from previous steps 
            pred_y = model(src.float(),src_mask=src_mask)
            loss = loss_function(pred_y, tgt.float())
            wandb.log({"training_loss": loss})
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item() 

        wandb.log({'total_training_loss':train_loss})

        model.eval()
        valid_loss = 0
        for src,tgt,id in validation_ldr:
            pred_y = model(src.float(),src_mask=src_mask)
            loss = loss_function(pred_y,tgt.float())
            valid_loss += loss.item()
            wandb.log({"validation_loss": loss})

    
        wandb.log({'total_valid_loss':valid_loss})
        hold_out_loss=0
        total_predictions = []
        total_targets = []
        for src,tgt,id in hold_out_ldr:
            pred_y = model(src.float(),src_mask=src_mask)
            loss = loss_function(pred_y,tgt.float())
            hold_out_loss += loss.item() 
            total_predictions.append(pred_y.detach().numpy())
            total_targets.append(tgt.detach().numpy())
            wandb.log({"hold_out_loss": loss})
            print(id[0])
            print(pred_y[0])
            print(tgt[0])
        epoch_time = time.time() - total_start
        wandb.log({'epoch time':epoch_time})

        T.save({
            'epoch': epoch_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'transformer_checkpoint.pt')





    wandb.log({'total_hold_out_loss':hold_out_loss})
    wandb.log({"hold_out_predictions": wandb.Histogram(total_predictions)})
    wandb.log({"hold_out_targets": wandb.Histogram(total_targets)})
    targets = np.concatenate(total_targets,axis=0)
    predictions =  np.concatenate(total_predictions,axis=0)
    print(targets)
    r2 = r2_score(targets,predictions)
    print('hold_out r2:',r2)
    #save model
    T.save(model.state_dict(), f'{cfg.path.project}/transformer_model.pt')
    
main()