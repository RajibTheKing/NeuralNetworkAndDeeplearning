"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

# ----------------------
# - read the input data:

import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

# ---------------------
# - network.py example:
import network2_mod
import matplotlib.pyplot as plt

numInputs = 784
numOutputs = 10
numHidden = 30
networkSize = [numInputs, numHidden, numOutputs]

learningRate = 0.05
numEpochs = 400 # instead of 30 as used before
batchSize = 30

# chapter 3 - Overfitting example - too many epochs of learning applied on small (1k samples) amount od data.
# Overfitting is treating noise as a signal.
net = network2_mod.Network(networkSize, cost=network2_mod.CrossEntropyCost)
net.large_weight_initializer()
eval_cost, eval_acc, train_cost, train_acc = net.SGD(training_data[:4500], numEpochs, batchSize, learningRate, evaluation_data=validation_data, 
                                                     monitor_training_accuracy=True, monitor_evaluation_accuracy=True,
                                                     monitor_training_cost=True, monitor_evaluation_cost=True)
  
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot( np.array([i+1 for i in range(numEpochs)]), train_acc, label = 'training acc.')
plt.plot( np.array([i+1 for i in range(numEpochs)]), eval_acc, label = 'validation acc.')
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right')
plt.title("training and validation accuracy")    
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot( np.array([i+1 for i in range(numEpochs)]), train_cost, label = 'training cost')
plt.plot( np.array([i+1 for i in range(numEpochs)]), eval_cost, label = 'validation cost')
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right')
plt.title("training and validation cost")    
plt.show()

test_acc = net.accuracy(test_data)

print("Learning rate = %f" %learningRate)
print("batch Size = %f" %batchSize)
print("test accuracy: %f" % (test_acc * 100.0 / len(test_data)))
