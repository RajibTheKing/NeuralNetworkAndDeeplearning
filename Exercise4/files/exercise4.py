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

training_data, validation_data, test_data = mnist_loader.load_data()
# NOTE that input data are normalized to the range [0;1]

# split into inputs and targets
training_input = training_data[0]
training_target = training_data[1]

validation_input = validation_data[0]
validation_target = validation_data[1]

test_input = test_data[0]
test_target = test_data[1]


# ---------------------
# - network.py example:
import numpy as np
from sklearn import neural_network as nn
import matplotlib.pyplot as plt

learningRate = 0.05   
numEpochs = 100
batchSize = 10 
numHiddenLayers = 5
numHiddenNeurons = 100
hiddenLayerSizes = ()
for k in range(numHiddenLayers):
    hiddenLayerSizes = hiddenLayerSizes + (numHiddenNeurons,)

print("number of hidden layers = %d" % numHiddenLayers)
print("hidden layer sizes:")
print(hiddenLayerSizes)


# The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.

#activation = 'logistic'
activation = 'relu'
net = nn.MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation=activation, solver='sgd', alpha=0.0, 
		                      batch_size=batchSize, learning_rate='constant', learning_rate_init=learningRate, 
							        max_iter=1, shuffle=False )

# train the network for one epoch
net.fit(training_input, training_target)	

# histogram of weights after one epoch
nBins = 100
fig = plt.figure()
plt.hist(net.coefs_[0].flatten(), nBins)
plt.xlabel("weights")
plt.ylabel("counts")
plt.title("weight histogram at beginning of training (after 1 epoch)")
plt.show()

# change number of epochs and continue training (warm start)
net.max_iter = numEpochs
net.warm_start = True
net.fit(training_input, training_target)	
	
# accuracies after training
train_acc = net.score(training_input, training_target)
print("training accuracy: %f" % train_acc)

eval_acc = net.score(validation_input, validation_target)
print("validation accuracy: %f" % eval_acc)
   
test_acc = net.score(test_input, test_target)	
print("test accuracy: %f" % test_acc)

# histogram of weights after training
nBins = 100
fig = plt.figure()
plt.hist(net.coefs_[0].flatten(), nBins)
plt.xlabel("weights")
plt.ylabel("counts")
plt.title("weight histogram after training")
plt.show()

# plot training loss curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar( np.array([i+1 for i in range(len(net.loss_curve_))]), net.loss_curve_, label = 'training loss')
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')
#    plt.xticks(range(1,numEpochs+1))
plt.title("training loss")    
plt.show()


# visualize the weights between input layer and some 
# of the hidden neurons of the first hidden layer 
# net.coefs_[0] is a (784 x numHiddenNeurons) array
# net.coefs_[0].T (transpose) is a (numHiddenNeurons x 784) array,
# the first entry of which contains the weights of all inputs connecting
# to the first hidden neuron; those weights will be displayed in (28 x 28) format
# until all plots (4 x 4, i.e. 16) are "filled" or no more hidden neurons are left
print("Visualization of the weights between input and some of the hidden neurons of the first hidden layer:")
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = net.coefs_[0].min(), net.coefs_[0].max()
for coef, ax in zip(net.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
