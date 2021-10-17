import torch


featureDim = 4
featureNames = ['volFrac','thetaX', 'thetaY', 'thetaZ']

labelDim = 9
labelNames = ['stiffness0', 'stiffness1', 'stiffness2', 'stiffness7', 'stiffness8', 'stiffness14', 'stiffness21', 'stiffness28', 'stiffness35'];

trainSplit = 0.9
testSplit = 0.1

batchSize = 64
batchShuffle = True
numWorkers = 0

randomWeights = False

fwdTrain = True
fwdEpochs =  200;
fwdHiddenDim = [128, 128, 64, 64, 32, 32]
fwdHiddenLayers = len(fwdHiddenDim)-1
fwdLearningRate = 1e-4
fwdLossFn = torch.nn.MSELoss()

invTrain = True
invEpochs =  200;
invHiddenDim = [100, 100, 100, 100, 100, 100]
invHiddenLayers = len(invHiddenDim)-1
invLearningRate = 1e-4
invLossFn = torch.nn.MSELoss()
betaX = 0.5
betaXEpochSchedule = 40
thetaMin = 0.1667 #normalized equivalent of 15 degrees
