
#script from https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py

# Usage:
# python -m torch.distributed.launch \
# --nproc_per_node <NUM_GPUS> \
# --nnodes 1 \
# --node_rank 0 \
# log-ddp.py \
# --log_all \
# --epochs 10 \
# --batch 512 \
# --entity <ENTITY> \
# --project <PROJECT>

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import hydra
from omegaconf import DictConfig
from lstm_model import RegressionLSTM
import os
import pandas as pd
from sklearn import preprocessing
from pickle import dump
from transformer.transformer_model import CMIPTimeSeriesDataset
import omegaconf
import time
def split_data(df):
    latlons = df[['lat','lon']].drop_duplicates()
    train = latlons.sample(frac=0.7)
    remainder = latlons.drop(train.index)
    val = remainder.sample(frac=0.66)
    test = remainder.drop(val.index)
    train = pd.merge(left=df,right=train,how='inner',on=['lat','lon'])
    val = pd.merge(left=df,right=val,how='inner',on=['lat','lon'])
    test = pd.merge(left=df,right=test,how='inner',on=['lat','lon'])
    return train,val,test

def get_training_data(cfg,run):
    cesm_df = pd.read_csv(f'{cfg.data}/timeseries_cesm_training_data_30.csv')
    #need to reshape here for data scaling
    # split data into 6 chunks, use 4 for training, 1 for validation, 1 for hold out
    # chunk_size = len((np.array_split(cesm_df, 6, axis=0))[0])
    # for var in cfg.model.output:
    #     cesm_df[var] = cesm_df[var].apply(lambda x: x*1000000000)
    # chunk_size = 12000
    # cesm_df = cesm_df[cfg.model.input + cfg.model.output + cfg.model.id]
    # cesm_df = cesm_df.to_numpy()
    # cesm_df = cesm_df.reshape(-1,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id))
    # np.random.shuffle(cesm_df)
    # cesm_df = cesm_df.reshape(-1,len(cfg.model.input + cfg.model.output + cfg.model.id))
    # cesm_df = pd.DataFrame(cesm_df)
    # cesm_df.columns = cfg.model.input + cfg.model.output + cfg.model.id
    # # hold_out = cesm_df[-chunk_size:]
    # # train_ds = cesm_df[:-chunk_size]
    # train_ds = cesm_df
    train_ds,val_ds,test_ds = split_data(cesm_df)
    #fix that you are modifying targets here too
    scaler = preprocessing.StandardScaler().fit(train_ds.loc[:,cfg.model.input])
    # hold_out_scaler = preprocessing.StandardScaler().fit(hold_out.loc[:,cfg.model.input])
    train_ds.loc[:,cfg.model.input] = scaler.transform(train_ds.loc[:,cfg.model.input])
    val_ds.loc[:,cfg.model.input] = scaler.transform(val_ds.loc[:,cfg.model.input])
    test_ds.loc[:,cfg.model.input] = scaler.transform(test_ds.loc[:,cfg.model.input])
    # train_ds.loc[:,cfg.model.output] = out_scaler.transform(train_ds.loc[:,cfg.model.output])
    # hold_out.loc[:,cfg.model.input] = scaler.transform(hold_out.loc[:,cfg.model.input])
    if run != None:
        dump(scaler, open(f'{cfg.environment.checkpoint}/lstm_scaler_{run.name}.pkl','wb'))
    # dump(hold_out_scaler, open(f'{cfg.environment.checkpoint}/lstm_hold_out_scaler_{wandb.run.name}.pkl','wb'))
    # hold_out = CMIPTimeSeriesDataset(hold_out,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    train_ds = CMIPTimeSeriesDataset(train_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    val_ds = CMIPTimeSeriesDataset(val_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    test_ds = CMIPTimeSeriesDataset(test_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)

    # train,validation = torch.utils.data.random_split(train_ds, [0.8,0.2], generator=torch.Generator().manual_seed(0))
    train_sampler = torch.utils.data.DistributedSampler(train_ds)
    validation_sampler = torch.utils.data.DistributedSampler(val_ds)
    test_sampler = torch.utils.data.DistributedSampler(test_ds)
    train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=cfg.model.batch_size,shuffle=False, sampler=train_sampler)#train sample shuffles for us
    validation_ldr = torch.utils.data.DataLoader(val_ds,batch_size=cfg.model.batch_size,shuffle=False,sampler=validation_sampler)
    test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=cfg.model.batch_size,shuffle=False,sampler=test_sampler)

    # hold_out_ldr = torch.utils.data.DataLoader(hold_out,batch_size=cfg.model.batch_size,shuffle=True)
    return train_sampler,train_ldr,validation_sampler,validation_ldr,test_sampler,test_ldr




def train(cfg, run=None):
    """
    Train method for the model.
    Args:
        args: The parsed argument object
        run: If logging, the wandb run object, otherwise None
    """
    # Check to see if local_rank is 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0
    do_log = run is not None

    # set the device
    

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend="gloo")

    # initialize model -- no changes from normal training
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=None, output_device=None
    )

    # watch gradients only for rank 0
    if is_master:
        run.watch(model)





    train_sampler,train_loader,validation_sampler,validation_loader,test_sampler,test_loader = get_training_data(cfg,run)

    total_step = len(train_loader)
    for epoch in range(cfg.model.epochs):
        total_start = time.time()
        batch_loss = []
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        model.train()
        for _,(src,tgt,_) in enumerate(train_loader):
            pred_y = model(src.float())
            loss = criterion(pred_y, tgt.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = float(loss)
            batch_loss.append(loss)
            if do_log:
                run.log({"train_batch_loss": loss})
        valid_batch_loss = []

        #Validation data
        model.eval()
        for _,(srd,tgt,_) in enumerate(validation_loader):
            pred_y = model(srd.float())
            loss = criterion(pred_y, tgt.float())
            loss = float(loss)
            valid_batch_loss.append(loss)
            if do_log:
                run.log({"valid_batch_loss": loss})
        test_batch_loss = []
        if do_log:
            run.log({"epoch": epoch, 'epoch_time':time.time() - total_start})
    #test data
    if is_master:
        for _,(srd,tgt,_) in enumerate(test_loader):
            pred_y = model(srd.float())
            loss = criterion(pred_y, tgt.float())
            loss = float(loss)
            test_batch_loss.append(loss)
            if do_log:
                run.log({"test_batch_loss": loss})
        if do_log:
            run.log({"epoch": epoch, "loss": np.mean(batch_loss),'valid_loss':np.mean(valid_batch_loss),'epoch_time':time.time() - total_start})


def setup_run(cfg):
    if os.environ['LOCAL_RANK'] == '0':
        run = wandb.init(entity=cfg.entity,project=cfg.wandb_project,config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    else:
        run = None
    print(run)
    print(os.environ['LOCAL_RANK'])
    return run

@hydra.main(version_base=None, config_path="conf",config_name='config')
def main(cfg: DictConfig):
    # wandb.init a run if logging, otherwise return None
    run = setup_run(cfg)

    train(cfg, run)


main()