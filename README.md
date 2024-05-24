# PROBLEM

- Model should predict the accuracy of a DNN model at a later stage
    - Report the accuracies of the eval_acc of CifarNet predicted by the accModel on the test dataset in the following settings: (E=5, M=150), (E=10,M=150), (E=20, M=150), (E=30, M=150), (E=60, M=150)

- Input:
    - hyperparameter vector Hi   
    - training and validation losss values and accuracy in the first E epochs
    - A positive integer M (m > e)


# DATASET

- CifarNet trained on Cifar 10
    - CifarNet contains 2 convolutional layers
    - Hyperparameters: L and H at each conv layer
    - Total 4 hyperparameter per model iteration

- HP_space.csv
    - sampled hyperparameter vector values

- train_loss.csv
    - First 4 calues -> Hyperparaeter for those iterations
    - Following columns show the training losses
    - 150 epochs recorded. Each epoch has 50 mini batches

- eval_loss.csv
    - Same as train_loss
    - Reports only 1 loss per epoch

- eval_acc.csv
    - same as above
    - Reports 1 accuracy per epoch

- dataPartition.txt
    - Use this as partitioning for training, testing and validating

# DATA MANIPULATION
- Find row mismatch (verified that eval_loss and eval_acc have no mismatch)
- Remove rows with Nans completely.
- Averaging loss over a batch stabilizes training by reducing noise from individual sample losses
    - Considering this, we can average over the 50 minibatches and get a consize train loss per epoch
- Combine the manipulated datasets into 1 csv
- Convert all hyerparameter values to integer
- Reduce the hyperparameters to be between 0 and 1 (divide by 10) -> To not allow feature overpowering and induce stability
- Create train, test and Validation Datasets

# DATA MANIPULATION CODE
- The code for data manipulation is in data_manipulation.ipynb
- The final data as a result of the manipulations described above is stored in the folder data
- three csv files are stored: train.csv, test.csv and val.csv
- Note that the partition.txt had to be slightly modified (values over 200 had to go).
    - This is because I found row mismatches between train_loss.csv and (eval_loss.csv | eval_train.csv)
    - I have fixed the issue and the explanation and code for the same is in data_manipulation.ipynb

# Modelling Strategy
- The data is a clear time series data
- I used 2 stacked layers of LSTM for prediction.
- The model size is pretty small because the data we have is very limited
- I create a new model for each E value.
- Model input:
    - vector of (Batch_size, time_steps, 7)
    - 7 because -> 4 hyper parameters + train_loss at epoch_i + eval_loss at epoch_i + eval_acc at epoch_i
    - (The hyper parameter vector is copied in each timestep as it doesn't change with time)
- Model output:
    - eval accuracy at epoch 150
- Loss function
    - MSE
- Optimizer
    - Adam

# Modelling code
- All the modelling code is in model.ipynb
- The models are checkpointed at every epoch
    - Saved models found at file path ./ckpt/e{E value}/model_epoch{epoch_number}.h5
- The model.ipynb also has code for generating training graphs and prediction graphs. (More in the PPT)

# Results
- Results are stored in the folder ./results/e{E Value}
- Depending on the E value, go to the appropriate folder
- inside the folder is a result.csv with predicted values for test_data for M=150 and the corresponding E value.

# Loss Logs
- Each model's train loss and val loss vectors are stored in a folder called "./model_final_loss_logs/e{E value}"

# Reproducibility
- Pip packages
    - pip install tensorflow==2.12.0
    - pip install matplotlib==3.7.1
    - pip install pandas==1.3.5
    - pip install numpy==1.22.4

- For training
    - Go through model.ipynb and run cells under the desired markdown heading

- For reproducing results
    - run "python reproduce_results.py"
    - The result will print on the screen
    - The result will also be in the folder ./reproduced_results/e{E value}

# A better approach?
- Instead of predicting just M=150, we can predict a vector of eval_accuracies.
    - Eg. predict accuracies from <M=70 to M=150>

- How to do it?
    - Many to Many LSTM

- Ideas worth exploring
    - Differnt models per E
    - A single model that can take “None” tokens for a dynamic E value.
    - Using small transformers (attention is all you need)

# PRESENTATION
- https://docs.google.com/presentation/d/11hrPQqH2pnBD6P9KZQhltc9O3mTAXaLUfK7KasWCnNE/edit?usp=sharing