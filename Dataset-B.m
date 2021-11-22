close all
clear all
clc


load humanactivity

rng('default') % For reproducibility
Partition = cvpartition(actid,'Holdout',0.10);
trainingInds = training(Partition); % Indices for the training set
XTrain = feat(trainingInds,:);
YTrain = actid(trainingInds);
testInds = test(Partition); % Indices for the test set
XTest = feat(testInds,:);
YTest = actid(testInds);


%Converting training Data to 1D Vector of 1x60 for training
for k=1:21668

 tempfeatures2=XTrain(k,:);

lstmnewTrainningFeature(k,:)=num2cell(double(tempfeatures2'),1:2);
k
end

%Converting testing Data to 1D Vector of 1x60 for testing
for k=1:2407

 tempfeatures3=XTest(k,:);

lstmnewTestedFeature(k,:)=num2cell(double(tempfeatures3'),1:2);
k
end
TrainningTargets = categorical(YTrain);

%% LSTM Trainning On original Features
inputSize = 60;
numHiddenUnits = 20;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = 50;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.005, ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


lstmnet = trainNetwork(lstmnewTrainningFeature,TrainningTargets,layers,options);

%% Classification And Accuracy On Test Data Set
LSTMPred = classify(lstmnet,lstmnewTestedFeature);
LSTMresult=double(LSTMPred);
actualtest1=double(YTest);
LSTMaccuracy = mean(LSTMresult == actualtest1);
disp('LSTM Accuracy On Testing Features');
LSTMaccuracy=LSTMaccuracy*100;



%% BILSTM Trainning On original Features
inputSize = 60;
numHiddenUnits = 20;
numClasses = 5;


layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = 30;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.005, ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


Bilstmnet = trainNetwork(lstmnewTrainningFeature,TrainningTargets,layers,options);

%% Classification And Accuracy On Test Data Set
biLSTMPred = classify(Bilstmnet,lstmnewTestedFeature);
biLSTMresult=double(biLSTMPred);
actualtest1=double(YTest);
biLSTMaccuracy = mean(biLSTMresult == actualtest1);
disp('BLSTM Accuracy On Testing Features');
biLSTMaccuracy=biLSTMaccuracy*100;


%% Trainning On CNN Network with original Features
% 0 Padding addition to training data at image edges
for i=1:21668
    XTrain(i,61:64)=0;
end

%Converting training 1D Data to 2D Matrix of 24x24 
for k=1:21668
    z=0;
    y=0;
    for j=1:8
        
        x=(y+1);
        y=(y+1)+7;
        a(j,:)=XTrain(k,x:y);
    end
    trainFourDArray(:,:,:,k)=a;
end

% 0 Padding adding to testing data at image edges
for i=1:2407
    XTest(i,61:64)=0;
end

%Converting testing 1D Data to 2D Matrix of 24x24 
for k=1:2407
    z=0;
    y=0;
    for j=1:8
        
        x=(y+1);
        y=(y+1)+7;
        b(j,:)=XTest(k,x:y);
    end
    TestFourDArray(:,:,:,k)=b;
end

imageSize = [8 8 1];
% %augimds = augmentedImageDatastore(imageSize,DArray,y_test);
% numHiddenUnits=64;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer([2 4],8,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    
    convolution2dLayer([2 8],18,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',1)
    
%     convolution2dLayer([2 18],36,'Padding','same')  
%     reluLayer
%     maxPooling2dLayer(2,'Stride',1)
%     
%     convolution2dLayer([2 36],72,'Padding','same')
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
%     convolution2dLayer([2 36],72,'Padding','same')
%     reluLayer
%     maxPooling2dLayer(2,'Stride',1)
%     dropoutLayer(0.5)

    fullyConnectedLayer(5)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam','MaxEpochs',25,...
    'MiniBatchSize',150,...
    'InitialLearnRate',0.002,'ExecutionEnvironment','gpu');
% % % options = trainingOptions('sgdm');
TrainningTargets = categorical(YTrain);
CNNnet = trainNetwork(trainFourDArray,TrainningTargets,layers,options);

%Accuracy
%% [YPred,probs]  = predict(net,TestFourDArray);

[YPred,probs]  = classify(CNNnet,TestFourDArray);

CNNresult=double(YPred);
actualtest=double(YTest);
CNNaccuracy = mean(CNNresult == actualtest);
disp('CNN Accuracy On Testing DataSet');
CNNaccuracy=CNNaccuracy*100

%% Parallel Model Accuracy
% checker=0;
for i=1:2407
    [c1,cp1] = classify(CNNnet,TestFourDArray(:,:,:,i));
    [l1,lp1] = classify(lstmnet,lstmnewTestedFeature(i));
    [b1,bp1] = classify(Bilstmnet,lstmnewTestedFeature(i));
    confidenceArray=[max(cp1) max(lp1) max(bp1)];
    catogoryArray=[c1 l1 b1];

cf1=cp1(c1)
cf2=lp1(c1)
cf3=bp1(c1)

lf1=cp1(l1)
lf2=lp1(l1)
lf3=bp1(l1)

bf1=cp1(b1)
bf2=lp1(b1)
bf3=bp1(b1)

cf=cf3+cf2+cf1
lf=lf3+lf2+lf1
bf=bf3+bf2+bf1
UnidentifiedConfidenceArray=[cf lf bf]

    [M,I] = max(UnidentifiedConfidenceArray);
    
    if(I==1)
        temresult(i,:)=c1;
    end
    if(I==2)
        temresult(i,:)=l1;
    end
    if(I==3)
        temresult(i,:)=b1;
    end
end
   checker=0;    
    i
% end

parallelresult=double(temresult);

ParallelMean = mean(parallelresult == actualtest1);
disp('Final Parallel Accuracy');
Parallelaccuracy=ParallelMean*100

%To check incorrect classification of instance
counter=1;
for g=1:2407
    
    if(parallelresult(g) ~= actualtest1(g))
       ParallelToActualActivity(counter,1)= g;
       ParallelToActualActivity(counter,2)=actualtest1(g);
       ParallelToActualActivity(counter,3)=parallelresult(g);
       counter=counter+1;
    end
end 

[pGC,pGR] = groupcounts(ParallelToActualActivity(:,3))

