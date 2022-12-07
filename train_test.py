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
# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() 
# and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement


if __name__ == "__main__":
   
    
    
    #prepare and scale data
    data = pd.read_csv(f'{DATA_PATH}/cesm_data_variant.csv')
    data = data.reset_index()
    data.columns = ['year','lat','lon','cOther','cCwd','cVeg','cLitter','cLeaf','cRoot','evspsblsoi','lai','tsl','mrro','mrsos','grassFrac','shrubFrac','cropFrac','baresoilFrac','residualFrac','treeFrac','shrubFrac','cSoil','cStem','wetlandFrac','ps','pr','tas_DJF','tas_JJA','tas_MAM','tas_SON','grassCropFrac','variant']

    data = data.rename(columns={'# year':'year'})
    print(data.size)
    ds = data[data['year'] < 2012]
    final_test = data[data['year'] >= 2012] 

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_scaler = preprocessing.StandardScaler()

    input_variables = constants.MODEL_INPUT_VARIABLES
    input_variable_tuple = tuple(constants.MODEL_INPUT_VARIABLES)
    target_variables = constants.MODEL_TARGET_VARIABLES
    target_variable_tuple = tuple(constants.MODEL_TARGET_VARIABLES)
    #split these so no data is leaked between test and training
    scaler = min_max_scaler.fit(ds.loc[:,input_variables])
    ds.loc[:,input_variable_tuple] = scaler.transform(ds.loc[:,input_variable_tuple])
    # ds.loc[:,target_variable_tuple] = min_max_scaler.fit_transform(ds.loc[:,target_variable_tuple])

    final_test.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(final_test.loc[:,input_variable_tuple])
    # final_test.loc[:,target_variable_tuple] = min_max_scaler.fit_transform(final_test.loc[:,target_variable_tuple])

    ds = CMIPDataset(ds[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))
    final_test = CMIPDataset(final_test[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables),num_of_targets=len(target_variables))

    #hyperparameters
    num_of_epochs = 200
    learning_rate = 0.0001
    test_validation_batch_size = 200

    #split train and validation
    train_set_size = int(len(ds) * 0.75)
    valid_set_size = len(ds) - train_set_size
    train,validation = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))
    # train.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(train.loc[:,input_variable_tuple])
    # validation.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(validation.loc[:,input_variable_tuple])
    model = Net(len(input_variables),len(target_variables))
    loss_function = nn.MSELoss()
    train_ldr = T.utils.data.DataLoader(train,batch_size=test_validation_batch_size,shuffle=True)
    validation_ldr = T.utils.data.DataLoader(validation,batch_size=test_validation_batch_size,shuffle=True)
    final_ldr = T.utils.data.DataLoader(final_test,batch_size=1,shuffle=False)
    optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []
    min_valid_loss = np.inf
    for epoch in range(num_of_epochs):
        start_time = time.time()
        train_loss = 0
        model.train()
        for X,y in train_ldr:
            optimizer.zero_grad() #clears old gradients from previous steps 

            pred_y = model(X.float())
            loss = loss_function(pred_y, y.float())
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
        training_losses.append(train_loss)


        model.eval()
        valid_loss = 0
        for X,y in validation_ldr:
            # print(X.data.numpy(),y.data.numpy())
            pred_y = model(X.float())
            # print(pred_y.data.numpy())
            test_loss = loss_function(pred_y,y.float())
            valid_loss += test_loss.item()
        validation_losses.append(valid_loss)
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
        print(f'Epoch finished in {time.time() - start_time} seconds')

    final_loss = 0
    total = 0
    pred_total = 0
    y_pred_list = []
    y_target = []
    for X,y in final_ldr:
        final_pred = model(X.float())
        loss = loss_function(final_pred,y.float())
        final_loss += loss.item() * X.size(0)
        total = total + y.float()
        pred_total += final_pred
        y_pred_list.append(final_pred.detach().numpy()[0])
        y_target.append(y.detach().numpy()[0])
    print(y_pred_list[0],y_target[0])
    # print(f'mse: {nn.functional.mse_loss(y_pred_list,y_target).item()}')
    print(f'total r2 score {r2_score(y_target,y_pred_list)}')
    print(f'individual r2 {R2Score(T.tensor(np.array(y_pred_list)),T.tensor(np.array(y_target)),multioutput="raw_values")}')
    print(final_loss,total,pred_total)
    T.save(model.state_dict(), '/Users/gclyne/thesis/data/trained_net')
    dump(scaler, open('/Users/gclyne/thesis/data/scaler.pkl', 'wb'))
    print(final_loss)
    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend()
    plt.title("Learning rate %f"%(learning_rate))
    plt.savefig('learning_rate_epoch.png')


    plt.scatter(y_target, y_pred_list)
    plt.xlabel('True Values [Z]')
    plt.ylabel('Predictions [Z]')
    # plt.legend()

    plt.savefig('pred_v_ref.png')




