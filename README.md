# Early Stop via Prediction for Hyperparameter Tuning

## Introduction

DNN model training takes time. In hyperparameter tuning (i.e., finding the best values for some hyperparameters for a DNN model), we would need to train the DNN model repeatedly on each of the hyperparameter values and run the test. It would take even more time.

The objective of this project is to see whether we can predict the final accuracy of a DNN model for given hyperparameters values, without training the DNN model completely. In another word, can we tell how good the hyperparamter values are by just observing the results in the first small number of training eporches of the DNN model? If we can do that, it can save a lot of time in hyperparameter tuning. 
 
The method we want to try in this project is to build a Machine Learning model (called accModel) to make the prediction. For a given DNN X and a set of k hyperparameters H=<h1, h2, ..., hk> to tune, the input to the accModel includes (i) a sample hyperparameter vector Hi=<h1i, h2i, ..., hki>, where hki is the sample value of the kth hyperparameter; (ii) the observed training and validation loss values and accuracy values in the first E eporchs; (iii) a positive integer M (M>E). The output of the accModel is the predicted validation accuracy of DNN X after M eporchs of training with Hi as the hyperparameter vector. 

## Dataset

The provided dataset were obtained by training **<u>CifarNet</u>** on the **Cifar-10** dataset. *CifarNet* contains two convolutional layers. There are two hyperparameters, ***L*** and ***H***, at each of the two convolutional layers. So there are four hyperparameters in the hyperparameter vector of *CifarNet*: <L1, H1, L2, H2>. 

* HP_space.csv: the sampled hyperparameter vector values. 

* train_loss.csv: the **training loss values**. The file is organized as follows:

  - Each row gives the observations on *CifarNet* on one sampled hyperparameter vector.  
  - The first 4 columns indicate the hyperparameter vector value (it's the same as in `HP_space.csv`). 
  - The following columns show the training losses. 150 epochs are recorded, and each epoch contains 50 mini-batches.

* eval_loss.csv: the **validation loss values** of *CifarNet*. It is in a similar format as `train_loss.csv`, but reports only one loss in each epoch.

* eval_acc.csv: the **validation accuracies** of *CifarNet*. It is in a similar format as `train_loss.csv`, but reports only one accuracy in each epoch.

* dataPartition.txt: the samples (i.e., line numbers in the above data files) to be used as training dataset, validation dataset, and test dataset for the development of accModel.

## Requirement

* Create one (or more) accModel that can achieve the objective
* Report the accuracies of the eval_acc of *CifarNet* predicted by the accModel on the test dataset in the following settings: (E=5, M=150), (E=10, M=150), (E=20, M=150), (E=30, M=150), (E=60, M=150). 
* Put your code in a Github repository. Organize it well with the necessary README, stating the content of the repository, instructions on how to install and run, and any known limitations. The repository should be made easy for others to reproduce your testing results without retraining the models, and at the same, include the necessary scripts and instructions for reproducing the training process if the users want to do that. 

## Q & A:
* What to submit? 
  - A PPT file clearly explaining your results and observations, and include the github link.
* What models to use?
  - No restrictions.
* What DNN frameworks to use? 
  - TensorFlow or PyTorch



