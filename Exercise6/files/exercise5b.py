import neurolab as nl
import numpy as np
import pylab as pl
import numpy.random as rand
import re

###-------------------
# format training data
###-------------------

# training set
input = np.loadtxt('exercise3a_input.txt')
numTrainingSamples = input.shape[0]
numInputNeurons    = input.shape[1] # here: 2 input neurons (since input patterns 2-dimensional) 


###-------------------------------------
# generate network (competitive network)
###-------------------------------------

numOutputNeurons = 4 # here: 4 output neurons (since 4 classes)

# create competitive network 
inputRange = [ [0.0, 1.0] for i in range(numInputNeurons) ]

# create network with 2 inputs and 4 neurons
net = nl.net.newc( inputRange, numOutputNeurons )

# Change train function
#net.trainf = nl.train.train_wta # for unknown reason, train_wta does not work


# print network information
# Number of network inputs:
print("number of network inputs: %d" % net.ci)

# Number of network outputs:
print("number of network outputs: %d" % net.co)

# Number of network layers:
print("number of network layers: %d" % len(net.layers))


# initialise network weights

print("network initialisation:\n")
# weights for neuron 1:
#net.layers[0].np['w'][0] = np.array([ ???, ??? ])
# weights for neuron 1:
#net.layers[0].np['w'][1] = np.array([ ???, ??? ])
# weights for neuron 1:
#net.layers[0].np['w'][2] = np.array([ ???, ??? ])
# weights for neuron 1:
#net.layers[0].np['w'][3] = np.array([ ???, ??? ])

print("weights after initialisation:") 
for i in range (numOutputNeurons):
    print("weights of output neuron %d: %s" % ( i, net.layers[0].np['w'][i] ))
                               

###---------------
# network training
###---------------

# network training
print("starting training\n")
numEpochs = 10
error = net.train(input, epochs=numEpochs, show=1)
print("training finished\n")

print("weights after training:") 
for i in range (numOutputNeurons):
    print("weights of output neuron %d: %s" % ( i, net.layers[0].np['w'][i] ))

# abbreviation for the weight vectors for later plotting:
# NOTE: each weight vector has two coordinates: 
#       one from input neuron 1 (x-coordinate), one from input neuron 2 (y-coordinate)
w1 = net.layers[0].np['w'][0] # first output neuron
w2 = net.layers[0].np['w'][1] # second output neuron
w3 = net.layers[0].np['w'][2] # third output neuron
w4 = net.layers[0].np['w'][3] # fourth output neuron
w = np.array([w1,w2,w3,w4])


###-----------
# plot results 
###-----------

class1 = np.loadtxt('exercise3a_class1.txt')
class2 = np.loadtxt('exercise3a_class2.txt')
class3 = np.loadtxt('exercise3a_class3.txt')
class4 = np.loadtxt('exercise3a_class4.txt')
center1 = np.loadtxt('exercise3a_center1.txt')
center2 = np.loadtxt('exercise3a_center2.txt')
center3 = np.loadtxt('exercise3a_center3.txt')
center4 = np.loadtxt('exercise3a_center4.txt')
center = np.array([center1, center2, center3, center4])

# plot results
import pylab as pl
pl.title('Classification with winner takes all')
pl.subplot(211)
pl.plot(error, label="training error")
pl.legend(['training error'])
pl.xlabel('Epoch number')
pl.ylabel('error (SAE - sum absolute error)')

pl.subplot(212)
pl.plot(input[:,0], input[:,1], '.', \
    center[:,0], center[:,1], 'yv', \
    w[:,0], w[:,1], 'p')  
pl.legend(['train samples', 'real centers', 'train centers'])
pl.show()

