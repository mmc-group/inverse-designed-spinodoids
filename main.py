# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pandas as pd
from parameters import *
from normalization import Normalization
from loadDataset import *
from model import *
from errorAnalysis import *

if __name__ == '__main__':

    torch.manual_seed(0)
    os.system('mkdir models')
    os.system('mkdir loss-history')
    
    ##############---FWD MODEL---##############
    fwdModel = createFNN(featureDim, fwdHiddenDim, fwdHiddenLayers, labelDim)
    fwdOptimizer = torch.optim.Adam(fwdModel.parameters(), lr=fwdLearningRate)    
    print('\n\n**************************************************************')
    print('fwdModel', fwdModel)
    print('**************************************************************\n')

    ##############---INV MODEL---##############
    invModel = createINN(labelDim, invHiddenDim, invHiddenLayers, featureDim)
    invOptimizer = torch.optim.Adam(invModel.parameters(), lr=invLearningRate)
    print('\n\n**************************************************************')
    print('invModel', invModel)
    print('**************************************************************\n')

    ##############---INIT DATA---##############
    train_set, test_set, featureNormalization = getDataset()
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize, shuffle=batchShuffle)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)

    ##############---Training---##############
    fwdEpochLoss = 0.0
    invEpochLoss = 0.0

    fwdTrainHistory = []
    fwdTestHistory = []
    invTrainHistory = []
    invTestHistory = []
    loader_all_train = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=len(train_set), shuffle=False)
    loader_all_test = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set), shuffle=False)
    x_all_train, y_all_train = next(iter(loader_all_train))
    x_all_test, y_all_test = next(iter(loader_all_test))

        
    if(fwdTrain):
        print('\nBeginning forward model training')
        print('-------------------------------------')
        ##############---FWD TRAINING---##############
        for fwdEpochIter in range(fwdEpochs):
            fwdEpochLoss = 0.0
            for iteration, batch in enumerate(train_data_loader, 0):
                #get batch
                x_train = batch[0]
                y_train = batch[1]
                #set train mode
                fwdModel.train()
                #predict
                y_train_pred = fwdModel(x_train)
                #compute loss
                fwdLoss = fwdLossFn(y_train_pred, y_train)
                #optimize
                fwdOptimizer.zero_grad()
                fwdLoss.backward()
                fwdOptimizer.step()
                #store loss
                fwdEpochLoss += fwdLoss.item()
            print(" {}:{}/{} | fwdEpochLoss: {:.2e} | invEpochLoss: {:.2e}".format(\
                "fwd",fwdEpochIter,fwdEpochs,fwdEpochLoss/len(train_data_loader),invEpochLoss/len(train_data_loader)))
            fwdTrainHistory.append(fwdLossFn(fwdModel(x_all_train),y_all_train).item())
            fwdTestHistory.append(fwdLossFn(fwdModel(x_all_test),y_all_test).item())
        print('-------------------------------------')
        #save model
        torch.save(fwdModel, "models/fwdModel.pt")
        #export loss history
        exportList('loss-history/fwdTrainHistory',fwdTrainHistory)
        exportList('loss-history/fwdTestHistory',fwdTestHistory)
    else:
        fwdModel = torch.load("models/fwdModel.pt")
        fwdModel.eval()

    if(invTrain):
        print('\nBeginning inverse model training')
        print('-------------------------------------')
        ##############---INV TRAINING---##############
        for invEpochIter in range(invEpochs):
            invEpochLoss = 0.0
            
            #Scheduling betaX:
            if(invEpochIter < betaXEpochSchedule):
                betaVal = betaX
            else:
                betaVal = 0
            
            for iteration, batch in enumerate(train_data_loader, 0):
                #get batch
                x_train = batch[0]
                y_train = batch[1]
                #set train mode
                invModel.train()
                #predict
                x_train_pred = invModel(y_train)
                y_train_pred_pred = fwdModel(x_train_pred)
                #compute loss
                invLoss =  invLossFn(y_train_pred_pred, y_train) + betaVal * invLossFn(x_train_pred, x_train)
                #optimize
                invOptimizer.zero_grad()
                invLoss.backward()
                invOptimizer.step()
                #store loss
                invEpochLoss += invLoss.item()
            print(" {}:{}/{} | betaX: {:.2e} | fwd EpochLoss: {:.2e} | invEpochLoss: {:.6e}".format(\
                "inv",invEpochIter,invEpochs, betaVal, fwdEpochLoss/len(train_data_loader),invEpochLoss/len(train_data_loader)))
            invTrainHistory.append(invLossFn(fwdModel(invModel(y_all_train)),y_all_train).item())
            invTestHistory.append(invLossFn(fwdModel(invModel(y_all_test)),y_all_test).item())
        print('-------------------------------------')
        #save model
        torch.save(invModel, "models/invModel.pt")
        #export loss history
        exportList('loss-history/invTrainHistory',invTrainHistory)
        exportList('loss-history/invTestHistory',invTestHistory)
    else:
        invModel = torch.load("models/invModel.pt")
        invModel.eval()

    #############---TESTING---##############
    x_test, y_test = next(iter(test_data_loader))
    
    with torch.no_grad():
        y_test_pred = fwdModel(x_test);
        x_test_pred = invModel(y_test);
        x_test_pred_uncorrected = x_test_pred.detach().clone()
        #fix values so that theta is not 0 or below thetaMin
        x_test_pred = correctionDirect(x_test_pred)
        y_test_pred_pred = fwdModel(x_test_pred);
    
        #############---POST PROC---##############
        print('\nR2 values:\n--------------------------------------------')
        print('Fwd test Y R2:',computeR2(y_test_pred, y_test),'\n')

        print('Inv test reconstruction Y R2:',computeR2(y_test_pred_pred, y_test),'\n')

        print('Inv test prediction X R2:',computeR2(x_test_pred, x_test))
        print('^^ Dont freak out; this is expected to be (very) low')
        print('--------------------------------------------\n')
        

    #############---EXAMPLE (post-training)---##############
    print('\n--------------------------------------------\n')
    print('EXAMPLE on how to use the fwd/inv-models')

    fwdModel = torch.load("models/fwdModel.pt")
    fwdModel.eval()

    invModel = torch.load("models/invModel.pt")
    invModel.eval()

    _, _, featureNormalization = getDataset()

    # Thetas and relative density chosen from the data file; 
    # Must have double square brackets:
    x_orig = torch.tensor([[0.621797,0.,66.9923,0.]]) 
    # Normalize to [0,1] for all paraemters
    x = featureNormalization.normalize(x_orig.clone());

    # Stiffnesses from line 2 of .csv file
    # Must have double square brackets
    y = torch.tensor([[0.5437044,0.15359520000000002,0.1766492,0.3811624,0.15203440000000001,0.535709,0.157053,0.1799434,0.158551]]) 

    y_pred = fwdModel(x)
    print('\nTrue stiffness in dataset: ',y.detach().numpy())
    print('Prediction from fwdModel : ',y_pred.detach().numpy())

    x_pred = invModel(y)
    #fix values so that theta is not 0 or below thetaMin
    x_pred = correctionDirect(x_pred.clone())
    # xhat is in [0,1] range for all parameters. Unnormalize to get them in proper range for rsepctive parameters.
    x_pred = featureNormalization.unnormalize(x_pred.clone())
    print('\nTrue parameters in dataset: ',x_orig.detach().numpy())
    print('Prediction from invModel  : ',x_pred.detach().numpy())
    print('(expected to be different from original due to ill-posed inverse problem)')

