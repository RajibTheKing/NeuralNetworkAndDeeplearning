import neurolab as nl
import numpy as np
import pylab as pl
import numpy.random as rand
import re

###-------------------
# format training data
###-------------------

# training set
input = np.loadtxt('digits_input.txt')
numTrainingSamples = input.shape[0]
numInputNeurons    = input.shape[1]


###-------------------------------------
# generate network (competitive network)
###-------------------------------------

numOutputNeurons = ??? # FIX!!!

# create competitive network 
inputRange = [ [0.0, 1.0] for i in range(numInputNeurons) ]

# create network with ??? inputs and ??? neurons
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


###---------------
# network training
###---------------

# network training
print("starting training\n")
numEpochs = 100
error = net.train(input, epochs=numEpochs, show=1)
print("training finished\n")

#print "weights after training:" 
#for i in range (numOutputNeurons):
#    print "weights of output neuron %d: %s" % ( i, net.layers[0].np['w'][i] )


###--------------------------
# test network after training
###--------------------------

# training set
print("test on training set:")
result = net.sim(input)
print(result)
print("")

## noisy test set
#print("test on independent, noisy test set:")
#input_noisy = np.loadtxt('digits_noisy_input.txt')
#result_noisy = net.sim(input_noisy)
#print(result_noisy)

###-----------
# plot results 
###-----------


# plot results
import pylab as pl
pl.title('Classification with winner takes all - digits')
pl.subplot(211)
pl.plot(error, label="training error")
pl.legend(['training error'])
pl.xlabel('Epoch number')
pl.ylabel('error (SAE - sum absolute error)')
pl.show()

