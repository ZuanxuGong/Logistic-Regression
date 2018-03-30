# -*- coding: utf-8 -*-
import string
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

file_features = ['trainingData.txt', 'validationData.txt', 'testData.txt']
file_labels = ['trainingLabels.txt', 'validationLabels.txt', 'testLabels.txt']

def generateData(file_feature, file_label):
    labels = [];
    for line in open(file_label):
        labels.append(int(line));
        
    features = [[] for i in range(len(labels))];
    count = 0;
    for line in open(file_feature):
        for x in line.split(','):
            features[count].append(int(x) / 255.0);
        count += 1;
    return features,labels

def lossCompute(features, labels, w):
    count = 0
    loss = 0
    for x in features:
        y = labels[count]
        predict = np.dot(w,x);
        e = math.exp(-y * np.dot(w,x))
        loss += math.log(1 + e)
        count += 1;
    return loss / len(labels);

#initial weight
w0 = []
for i in range(784):
    w0.append(0) 

#get training + validation + testing data (features + labels)
[featuresTrain, labelsTrain] = generateData(file_features[0], file_labels[0]);
[featuresValidation, labelsValidation] = generateData(file_features[1], file_labels[1]);
[featuresTest, labelsTest] = generateData(file_features[2], file_labels[2]);

mius = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3, 3, 10]    
#cross-validation
for miu in mius:
    w = w0    
    for iter in range(1000):
        fwdao = 0
        count = 0;
        for x in featuresTrain:
            y = labelsTrain[count]
            e = math.exp(-y * np.dot(w,x))
            fwdao += (-y * np.array(x) * e) / (1 + e)
            count += 1;
        w = w - miu * fwdao / 6107;
        sys.stdout.write(' ' * 10 + '\r')
        sys.stdout.flush()
        sys.stdout.write("Iteration:" + str(iter) + '/' + str(1000) +'\r')
        sys.stdout.flush()
    print("learning rate u = %f " % (miu))
    #check training + validation performance
    lossTrain = lossCompute(featuresTrain, labelsTrain, w)
    lossValidation = lossCompute(featuresValidation, labelsValidation, w)
    print("training loss: %f  validation loss: %f" % (lossTrain, lossValidation))

#best miu = 1
miu = 1
w = w0
lossTrain = []
lossValidation = []
for iter in range(1000):
    fwdao = 0
    count = 0;
    for x in featuresTrain:
        y = labelsTrain[count]
        e = math.exp(-y * np.dot(w,x))
        fwdao += (-y * np.array(x) * e) / (1 + e)
        count += 1;
    w = w - miu * fwdao / 6107;
    sys.stdout.write(' ' * 10 + '\r')
    sys.stdout.flush()
    sys.stdout.write("Iteration:" + str(iter) + '/' + str(1000) +'\r')
    sys.stdout.flush()
    #check training + validation performance
    lossTrain.append(lossCompute(featuresTrain, labelsTrain, w))
    lossValidation.append(lossCompute(featuresValidation, labelsValidation, w))

iterations = np.linspace(1, 1000, 1000)
plt.subplot(111)
plt.plot(iterations, np.array(lossTrain), label = "training risk")
plt.plot(iterations, np.array(lossValidation), label = "validation risk")
plt.axis([1,1000,0,1])
plt.title('empirical risk')
plt.grid(True)
plt.legend()
plt.show()
    
print("best learning rate u = %f " % (miu))
#check training + validation performance
lossTrainBest = lossCompute(featuresTrain, labelsTrain, w)
lossValidationBest = lossCompute(featuresValidation, labelsValidation, w)
print("training error: %f  validation error: %f" % (lossTrainBest, lossValidationBest))    
lossTestBest = lossCompute(featuresTest, labelsTest, w)
print("testing error: %f" % (lossTestBest))