#generate sliding window timeseries of 30 years for each grid cell? 
import pandas as pd
import numpy as np
import torch as T
from tqdm import tqdm
from transformer_no_decoder import TestTransformer,generate_square_subsequent_mask
from other.constants import MODEL_INPUT_VARIABLES, MODEL_TARGET_VARIABLES

import wandb

wandb.init(project="test-project", entity="gclyne")


ds = T.load('data.pt')
num_of_epochs =200
#split train and validation
test_validation_batch_size=128
learning_rate = 0.0001
train_set_size = int(len(ds) * 0.5)
valid_set_size = int(len(ds) * 0.5)

# Input length
enc_seq_len = 30

T.set_printoptions(sci_mode=False)

train,validation = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(2))

model = TestTransformer(
    input_size=len(MODEL_INPUT_VARIABLES),
    dec_seq_len=30,
    batch_first=True,
    dim_val=512,
    n_encoder_layers=3,
    n_heads=16,
    num_predicted_features=len(MODEL_TARGET_VARIABLES)
)

for p in model.parameters():
    if p.dim() > 1:
        T.nn.init.xavier_uniform_(p.data)

# train = T.utils.data.Subset(train,range(0,test_validation_batch_size))
# validation = T.utils.data.Subset(validation,range(0,test_validation_batch_size))
loss_function = T.nn.MSELoss()
train_ldr = T.utils.data.DataLoader(train,batch_size=test_validation_batch_size,shuffle=True)
validation_ldr = T.utils.data.DataLoader(validation,batch_size=test_validation_batch_size,shuffle=True)

optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
training_losses = []
validation_losses = []
min_valid_loss = np.inf
src_mask = generate_square_subsequent_mask(
    dim1=enc_seq_len,
    dim2=enc_seq_len
    )


for epoch in range(num_of_epochs):
    train_loss = 0
    model.train()
    for src,tgt in tqdm(train_ldr):
        print(tgt)
        optimizer.zero_grad() #clears old gradients from previous steps 
        pred_y = model(src.float(),src_mask=src_mask)
        print('training predicton: ',pred_y)
        loss = loss_function(pred_y, tgt.float())
        wandb.log({"training_loss": loss})
        loss.backward() #compute gradient
        optimizer.step() #take step based on gradient
        train_loss += loss.item() / test_validation_batch_size

    wandb.log({'avg_training_loss':train_loss})

    model.eval()
    valid_loss = 0
    for src,tgt in tqdm(validation_ldr):
        pred_y = model(src.float(),src_mask=src_mask)
        loss = loss_function(pred_y,tgt.float())
        valid_loss += loss.item() / test_validation_batch_size
        wandb.log({"validation_loss": loss})
        print('validation prediction: ',pred_y)

 
    wandb.log({'avg_valid_loss':valid_loss})
