close all
clear all %#ok<CLALL>
clc
load('X_test.txt');
load('y_test.txt');
load('X_train.txt');
load('y_train.txt');


%Converting training Data to 1D Vector of 1x561 for training
for k=1:7767

 tempfeatures2=X_train(k,:);

lstmnewTrainningFeature(k,:)=num2cell(double(tempfeatures2'),1:2);
k
end


%Converting testing Data to 1D Vector of 1x561 for testing
for k=1:3162

 tempfeatures3=X_test(k,:);

lstmnewTestedFeature(k,:)=num2cell(double(tempfeatures3'),1:2);
k
end
TrainningTargets = categorical(y_train);


%% LSTM Trainning On original Features
inputSize = 561;
numHiddenUnits = 150;
numClasses = 12;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer]

maxEpochs = 400;
miniBatchSize = 150;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.002, ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


lstmnet = trainNetwork(lstmnewTrainningFeature,TrainningTargets,layers,options);


%% Classification Accuracy
LSTMPred = classify(lstmnet,lstmnewTestedFeature);
LSTMresult=double(LSTMPred);
actualtest1=double(y_test);
LSTMaccuracy = mean(LSTMresult == actualtest1);
disp('LSTM Accuracy On Testing Features');
LSTMaccuracy=LSTMaccuracy*100


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% BILSTM Trainning On original Features
inputSize = 561;
numHiddenUnits = 100;
numClasses = 12;


layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer]

maxEpochs = 400;
miniBatchSize = 150;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.002, ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


Bilstmnet = trainNetwork(lstmnewTrainningFeature,TrainningTargets,layers,options);

%% Classification Accuracy
biLSTMPred = classify(Bilstmnet,lstmnewTestedFeature);
biLSTMresult=double(biLSTMPred);
actualtest1=double(y_test);
biLSTMaccuracy = mean(biLSTMresult == actualtest1);
disp('BLSTM Accuracy On Testing Features');
biLSTMaccuracy=biLSTMaccuracy*100


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Trainning On CNN Network with original Features

% 0 Padding addition to training data at image edges
for i=1:7767
    X_train(i,562:576)=0;
end

%Converting training 1D Data to 2D Matrix of 24x24 
for k=1:7767
    z=0;
    y=0;
    for j=1:24
        
        x=(y+1);
        y=(y+1)+23;
        a(j,:)=X_train(k,x:y);
    end
    trainFourDArray(:,:,:,k)=a;
end

% 0 Padding adding to testing data at image edges
for i=1:3162
    X_test(i,562:576)=0;
end

%Converting testing 1D Data to 2D Matrix of 24x24 
for k=1:3162
    z=0;
    y=0;
    for j=1:24
        
        x=(y+1);
        y=(y+1)+23;
        b(j,:)=X_test(k,x:y);
    end
    TestFourDArray(:,:,:,k)=b;
end

imageSize = [24 24 1];

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

    fullyConnectedLayer(12)
    batchNormalizationLayer
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam','MaxEpochs',50,...
    'MiniBatchSize',150,...
    'InitialLearnRate',0.002,'ExecutionEnvironment','gpu');

TrainningTargets = categorical(y_train);
CNNnet = trainNetwork(trainFourDArray,TrainningTargets,layers,options);

%% Classification Accuracy
[YPred,probs]  = classify(CNNnet,TestFourDArray);
CNNresult=double(YPred);
actualtest=double(y_test);
CNNaccuracy = mean(CNNresult == actualtest);
disp('CNN Accuracy On Testing DataSet');
CNNaccuracy=CNNaccuracy*100


%% Decision Fusion

% checker=0;
for i=1:3162
    [c1,cp1] = classify(CNNnet,TestFourDArray(:,:,:,i));
    [l1,lp1] = classify(lstmnet,lstmnewTestedFeature(i));
    [b1,bp1] = classify(Bilstmnet,lstmnewTestedFeature(i));
    confidenceArray=[max(cp1) max(lp1) max(bp1)];
    catogoryArray=[c1 l1 b1];
% 
% if(c1==l1)
%     temresult(i,:)=c1;
%     checker=1;
% end
% 
% if(c1==b1)
%     temresult(i,:)=c1;
%     checker=1;
% end
% if(l1==b1)
%     temresult(i,:)=b1;
%     checker=1;
% end
% if(checker==0)
%     
% %%%%
%     [M,I] = max(confidenceArray);
%     
%     if(I==1)
%         temresult(i,:)=c1;
%     end
%     if(I==2)
%         temresult(i,:)=l1;
%     end
%     if(I==3)
%         temresult(i,:)=b1;
%     end

cf1=cp1(c1);
cf2=lp1(c1);
cf3=bp1(c1);

lf1=cp1(l1);
lf2=lp1(l1);
lf3=bp1(l1);

bf1=cp1(b1);
bf2=lp1(b1);
bf3=bp1(b1);

cf=cf3+cf2+cf1;
lf=lf3+lf2+lf1;
bf=bf3+bf2+bf1;

UnidentifiedConfidenceArray=[cf lf bf];

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
%    checker=0;    
%     i
% % end

parallelresult=double(temresult);
ParallelMean = mean(parallelresult == actualtest1);
disp('Final Model Accuracy');
Parallelaccuracy=ParallelMean*100;


%% To check incorrect Classification of Activities
counter=1;
for g=1:3162
    
    if(parallelresult(g) ~= actualtest1(g))
       ParallelToActualActivity(counter,1)= g;
       ParallelToActualActivity(counter,2)=actualtest1(g);
       ParallelToActualActivity(counter,3)=parallelresult(g);
       counter=counter+1;
    end
end 

[pGC,pGR] = groupcounts(ParallelToActualActivity(:,3))
