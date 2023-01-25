from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from transformer_model import TimeSeriesTransformer,CMIPTimeSeriesDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn import preprocessing
# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size,fn,cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    fn(rank, world_size,cfg)

def cleanup():
    dist.destroy_process_group()
    print('cleanup finished')

from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0,cfg=None):
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
    train,validation = torch.utils.data.random_split(train_ds, [int((chunk_size/30)*4), int(chunk_size/30)], generator=torch.Generator().manual_seed(0))

    sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = torch.DataLoader(train, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def demo_basic(rank, world_size,cfg):
    print(f"Running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
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
                   ).to(f'cpu:{rank}')
    ddp_model = DDP(model)
    print('model created')
    for i in range(50):
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(f'cpu:{rank}')
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss,i)
    cleanup()

def train(rank, world_size,model):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = model.to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously
   
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(10):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        

        for src,tgt,id in dataloader:
            optimizer.zero_grad() #clears old gradients from previous steps 
            pred_y = model(src.float())
            loss = loss_fn(pred_y, tgt.float())
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item() 
            print(train_loss)
    cleanup()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    size = 2
    processes = []
    # mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=setup, args=(rank, size, train,cfg))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

main()