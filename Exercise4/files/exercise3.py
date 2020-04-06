
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

numEpochs = 30
batchSize = 10
learningRates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15]
numRepetitions = 4
hiddenLayerSizes = (100)

optLearningRate = 0
optValidationAccuracy = 0

train_accuracy = np.zeros(numRepetitions)
validation_accuracy = np.zeros(numRepetitions)
test_accuracy = np.zeros(numRepetitions)

mean_train_accuracy = np.zeros(len(learningRates))
std_train_accuracy = np.zeros(len(learningRates))
mean_eval_accuracy = np.zeros(len(learningRates))
std_eval_accuracy = np.zeros(len(learningRates))
mean_test_accuracy = np.zeros(len(learningRates))
std_test_accuracy = np.zeros(len(learningRates))

for k in range(len(learningRates)):

    print("MODIFYING LEARNING RATE")
    learningRate = learningRates[k]
    print("learning rate = %f" % learningRate)
	
    train_loss = np.zeros((numRepetitions, numEpochs)) 
    mean_train_loss = np.zeros(numEpochs)
    std_train_loss = np.zeros(numEpochs)
    train_acc = np.zeros(numRepetitions)
    eval_acc = np.zeros(numRepetitions)
    test_acc = np.zeros(numRepetitions)
    
    for i in range(numRepetitions):
        print("Iteration %d..." % i)
        net = nn.MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, activation='logistic', solver='sgd', alpha=0.0, 
		                       batch_size=batchSize, learning_rate='constant', learning_rate_init=learningRate, 
							   max_iter=numEpochs, shuffle=False )
        net.fit(training_input, training_target)				  
        # loss curve contained in attribute loss_curve_ 
        for j in range(len(net.loss_curve_)): # NOT numEpochs due to potential early stopping
            train_loss[i][j] = net.loss_curve_[j]
		
		   # note that the training accuracy is not logged over the iterations;
	   	# so only the final accuracy at the end of iterations is available via the "score" method
        # (to monitor the training / validation / test accuracy, one might use the "warm_start" option)   
        train_acc[i] = net.score(training_input, training_target)
#        print("training accuracy: %f" % train_acc[i])
        
        eval_acc[i] = net.score(validation_input, validation_target)
#        print("validation accuracy: %f" % eval_acc[i])
           
        test_acc[i] = net.score(test_input, test_target)	
#        print("test accuracy: %f" % test_acc[i])

		
    mean_train_accuracy[k] = train_acc.mean()
    std_train_accuracy[k] = train_acc.std()
    mean_eval_accuracy[k] = eval_acc.mean()
    std_eval_accuracy[k] = eval_acc.std()
    mean_test_accuracy[k] = test_acc.mean()
    std_test_accuracy[k] = test_acc.std()
    
    # determine optimal learning rate
    if mean_eval_accuracy[k] > optValidationAccuracy:
        optValidationAccuracy = mean_eval_accuracy[k]
        optLearningRate = learningRate    

    # print results:
    print("training accuracy (in brackets: mean +/- std):")
    for i in range(numRepetitions):
        print("%f" % train_acc[i])
    print("(%f +/- %f)\n" % (mean_train_accuracy[k], std_train_accuracy[k]))
    
    print("validation accuracy (in brackets: mean +/- std):")
    for i in range(numRepetitions):
        print("%f" % eval_acc[i])
    print("(%f +/- %f)\n" % (mean_eval_accuracy[k], std_eval_accuracy[k]))
    
    print("test accuracy (in brackets: mean +/- std):")
    for i in range(numRepetitions):
        print("%f" % test_acc[i])
    print("(%f +/- %f)\n" % (mean_test_accuracy[k], std_test_accuracy[k]))

    # calculate and plot mean loss
    mean_train_loss = train_loss.mean(axis=0)
    std_train_loss = train_loss.std(axis=0)	
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar( np.array([i+1 for i in range(numEpochs)]), mean_train_loss, yerr=std_train_loss, label = 'training loss')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
#    plt.xticks(range(1,numEpochs+1))
    plt.title("training loss")    
    plt.show()

# plot training, validation and test accuracy as a function of the learning rate    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(learningRates, mean_train_accuracy, yerr=std_train_accuracy, label = 'training accuracy')
ax.errorbar(learningRates, mean_eval_accuracy, yerr=std_eval_accuracy, label = 'validation accuracy')
ax.errorbar(learningRates, mean_test_accuracy, yerr=std_test_accuracy, label = 'test accuracy')
ax.set_xlabel("learning rate")
ax.set_ylabel("accuracy")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left')
plt.title("mean accuracy as function of the learning rate")
plt.show()

# print test accuracy as function of learning rate
print("learning rates tested: %s" % learningRates)
print("mean training accuracy: %s" % mean_train_accuracy)
print("mean validation accuracy: %s" % mean_eval_accuracy)
print("mean test accuracy: %s" % mean_test_accuracy)
print("\n")

# print optimal learning rate
print("optimal learning rate (determined on validation data): %f" % optLearningRate)
