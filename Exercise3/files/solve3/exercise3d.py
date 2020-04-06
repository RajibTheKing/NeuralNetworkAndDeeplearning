from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

plot_args = [{'c': 'black', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]

###-----------------
# load training data
###-----------------

input = np.loadtxt('exercise3b_input.txt')
tmp = np.loadtxt('exercise3b_target.txt')



target = np.array([tmp[i] for i in range(tmp.size)])
print(target)
class1 = np.loadtxt('exercise3b_class1.txt')
class2 = np.loadtxt('exercise3b_class2.txt')

###-----------------------------------------
# generate network (single-layer perceptron)
###-----------------------------------------
 
numEpochs = 100 # FIX!!!

net = MLPClassifier(activation='logistic', hidden_layer_sizes=(), batch_size=1, 
                    learning_rate='constant', learning_rate_init=0.1,
                    shuffle=False, max_iter=1, warm_start=True, momentum = 0.0,
                    nesterovs_momentum=False, solver='sgd', verbose=True,
                    validation_fraction=0.0, alpha=0.0, tol=0.00001) # default for tol: 0.0001

###--------
# plot data 
###--------

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
legend = []
axes[0].set_title('Toy classification problem: Data and decision boundaries')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
minx = min(input[:,0])
maxx = max(input[:,0])
miny = min(input[:,1])
maxy = max(input[:,1])

axes[0].set_xlim(minx, maxx)
axes[0].set_ylim(miny, maxy) 
axes[0].plot(class1[:,0], class1[:,1], 'r.', \
    class2[:,0], class2[:,1], 'b.')
legend.append('samples class1')
legend.append('samples class2')

###---------------
# network training
###---------------
print("starting training\n")

idx = 0 # index to generate intermediate decision boundaries
colors = ['black', 'red', 'green', 'blue', 'magenta', 'cyan', 'yellow' ]

for i in range(numEpochs):
    net.fit(input, target) 
    if (i % 20 == 0):
        # network parameters: 
        # bias (list of numpy array)
        w0 = net.intercepts_[0][0]

        # weights (list of of numpy arrays of shape n_in x n_out)
        w1 = net.coefs_[0][0][0]
        w2 = net.coefs_[0][1][0]
        if ( w2 == 0 ):
            print("Error: second weight zero!")
            
        # plot intermediate decision boundary     
        interval = np.arange( np.floor(minx), np.ceil(maxx), 0.1 )
        line = -w1*interval/w2 - w0/w2
        args = {'c': 'black', 'linestyle': '--'}
        axes[0].plot( interval, line, linestyle='--', color=colors[idx])
        idx = idx+1 # for next plot
        legend.append('after iteration ' + str(i+1))


# weights and intercepts
print("weights after training:") 
print(net.coefs_)
print("Bias after training:")
print(net.intercepts_)

# network parameters: 
# bias (list of numpy array)
w0 = net.intercepts_[0][0]

# weights (list of of numpy arrays of shape n_in x n_out)
w1 = net.coefs_[0][0][0]
w2 = net.coefs_[0][1][0]
if ( w2 == 0 ):
    print("Error: second weight zero!")

# plot last decision boundary
interval = np.arange( np.floor(minx), np.ceil(maxx), 0.1 )
line = -w1*interval/w2 - w0/w2
args = {'c': 'black', 'linestyle': '-'}
axes[0].plot( interval, line, **args)
legend.append('after last iteration')
fig.legend(axes[0].get_lines(), legend, ncol=3, loc="upper center")
print("training finished\n")


###------------------------
# print network information
###------------------------
# Number of network inputs:
print("number of network inputs: %d" % len(input[0]))

# Number of network outputs:
print("number of network outputs: %d" % net.n_outputs_)

# Number of network layers:
print("number of network layers: %d" % net.n_layers_)

# count number of binary errors
error_binary = 0
result = net.predict( input )
result_binary = 0.5 * ( np.sign(result-0.5) + 1 )
# for counting binary errors: no factor 0.5 (as in SSE)
error_binary = np.sum( ( target - result_binary ) * ( target - result_binary ) )
print("\nnumber of binary errors: %d" % error_binary)

###--------------
# plot loss curve 
###--------------

axes[1].plot(net.loss_curve_)
axes[1].set_title('Toy classification problem: Loss curve')
axes[1].set_xlabel('Epoch number')
axes[1].set_ylabel('loss')
plt.show()

