#generate sliding window timeseries of 30 years for each grid cell? 
import pandas as pd
import numpy as np
from other.utils import readCoordinates
import torch as T
from tqdm import tqdm
from time_series_transformer import TimeSeriesTransformer,generate_square_subsequent_mask,CMIPTimeSeriesDataset
from sklearn.metrics import r2_score
from torchmetrics.functional import r2_score as R2Score
from other.constants import MODEL_INPUT_VARIABLES, MODEL_TARGET_VARIABLES

import wandb

wandb.init(project="test-project", entity="gclyne")

cesm_data = pd.read_csv('/Users/gclyne/thesis/data/cesm_data.csv')
managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)

wandb.config = {
  "learning_rate": 0.00001,
  "epochs": 10,
  "batch_size": 200
}


ds = CMIPTimeSeriesDataset(cesm_data,managed_forest_coordinates)
num_of_epochs = 30
#split train and validation
test_validation_batch_size=200
learning_rate = 0.00001
train_set_size = int(len(ds) * 0.75)
valid_set_size = int(len(ds) * 0.2)
hold_out = int(len(ds) * .05) + 1
# Input length
enc_seq_len = 30
# Output length
output_sequence_length = 5


train,validation,hold_out = T.utils.data.random_split(ds, [train_set_size, valid_set_size,hold_out], generator=T.Generator().manual_seed(0))

model = TimeSeriesTransformer(
    input_size=len(MODEL_INPUT_VARIABLES),
    dec_seq_len=30,
    batch_first=True,
    dim_val=24,
    n_encoder_layers=64,
    n_decoder_layers=64,
    n_heads=8,
    num_predicted_features=len(MODEL_TARGET_VARIABLES)
)

loss_function = T.nn.MSELoss()
train_ldr = T.utils.data.DataLoader(train,batch_size=test_validation_batch_size,shuffle=True)
validation_ldr = T.utils.data.DataLoader(validation,batch_size=test_validation_batch_size,shuffle=True)
hold_out_loader = T.utils.data.DataLoader(hold_out,batch_size=10,shuffle=True)

optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
training_losses = []
validation_losses = []
min_valid_loss = np.inf



# Make src mask for decoder with size:
tgt_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=output_sequence_length
   )

src_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    )


for epoch in range(num_of_epochs):
    train_loss = 0
    model.train()
    for src,tgt,tgt_y in tqdm(train_ldr):
        optimizer.zero_grad() #clears old gradients from previous steps 
        pred_y = model(src.float(),tgt.float(),src_mask=src_mask,tgt_mask=tgt_mask)
        loss = loss_function(pred_y, tgt_y.float())
        wandb.log({"training_loss": loss})
        wandb.watch(model)

        loss.backward() #compute gradient
        optimizer.step() #take step based on gradient
        train_loss += loss.item() / src.size(0)


    model.eval()
    valid_loss = 0
    for src,tgt,tgt_y in tqdm(validation_ldr):
        # print(X.data.numpy(),y.data.numpy())
        pred_y = model(src.float(),tgt.float(),src_mask=src_mask,tgt_mask=tgt_mask)
        # print(pred_y.data.numpy())
        loss = loss_function(pred_y,tgt_y.float())
        valid_loss += loss.item() / src.size(0)
        wandb.log({"validation_loss": loss})

    wandb.log({'avg_valid_loss':valid_loss,'avg_training_loss':train_loss})

hold_out_pred_list = []
hold_out_target = []

for X,y in hold_out_loader:
    final_pred = model(X.float(),y.float())
    loss = loss_function(final_pred,y.float())
    hold_out_pred_list.append(final_pred.detach().numpy()[0])
    hold_out_target.append(y.detach().numpy()[0])
print(hold_out_pred_list[0],hold_out_target[0])
print(f'total r2 score {r2_score(hold_out_target,hold_out_pred_list)}')
print(f'individual r2 {R2Score(T.tensor(np.array(hold_out_pred_list)),T.tensor(np.array(hold_out_target)),multioutput="raw_values")}')
