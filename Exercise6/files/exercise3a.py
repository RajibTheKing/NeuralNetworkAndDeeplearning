# adapted from https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# see also https://keras.io/getting-started/sequential-model-guide/
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
#from keras.layers import Input
#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

input_dim = 3      # dimensionality of input space
num_lstm_units = 2 # number of hidden units 
output_dim = 4     # dimensionality of output space
num_time_steps = 4 # number of time steps in sequence presented to the network
num_samples = 1    # number of samples e.g. in training (here not relevant, therefore set to 1)

# first possibility to define the model
#inputs1 = Input(shape=(num_time_steps, feature_dim)) 
#lstm1, state_h, state_c = LSTM(num_lstm_units, return_sequences = True, return_state = True)(inputs1)
#model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# second possibility to define the model
model = Sequential()
model.add(LSTM(
        input_dim=input_dim,
        output_dim=num_lstm_units,
        return_sequences=True)) # with return_sequences=True, an output will be generated for EACH input
                                # with return_sequences=False, only the last output will be generated 
                                # (corresponding to the last output with return_sequences=True)
                                # with the additional argument return_state = True, the cell state is returned;
                                # however, this seems to only work if the output layer is removed
model.add(Dense(
    output_dim=output_dim))
model.add(Activation("relu"))

w = [] # LSTM weights in Keras
# LSTM weights are in list format, see e.g. model.weights or model.get_weights()
# w[0] are the weights between input and hidden layer
# w[1] are the recurrent weights within the hidden layer
# w[2] are the biases 
# all parameters are stored in the format [input, forget, cell, output], where input, forget,
# cell and output refer to the values for ALL LSTM units
# e.g. w[2]=np.array([1,2,0,0,-2,-1,-0.5,1.5]): b_i = [1,2], b_f = [0,0], b_c = [-2, -1], b_o = [-0.5, 1.5]
#w.append(np.zeros((feature_dim, 4*num_lstm_units))) # weights between input and hidden layer
#w.append(np.zeros((num_lstm_units, 4 * num_lstm_units))) # recurrent weights within hidden layer
#w.append(np.random.random_sample((4 * num_lstm_units))) # bias values (factor 4 for input, forget, cell and output)

## specific example (2 LSTM units, 3 features)

# weights from input to the gates 
w.append([[2,4,1,2,1,0,-1,0],[0,1,0,1,-1,2,2,1], [3,0,-1,2,0,1,-2,1]])
# recurrent weights
w.append([[-3,-5,-2,2,2,-1,1,0],[0,1,0,1,0,1,1,2]])
# hidden biases
w.append([0, -2, 1, -1, -1, 2, -1, -1])
# weights from hidden activations to output
w.append([[2, 0, 1, 4],[-1, 3, -1, 0]])
# output biases
w.append([2, 0, -1, 1])
model.set_weights(w)
print("model weight summary: %s" % model.weights)
print("\n")
print("model weights: %s" % model.get_weights())
print("\n")

#x = np.random.random_sample((num_samples, num_time_steps, feature_dim))
#x = np.zeros((num_samples, num_time_steps, feature_dim))
x = np.array([[[1, 0, -1], [0, 1, 1], [-1, 1, 1], [1, 0, 0]]]) # shape (1, 4, 3)
print("input: %s" % x)
print("output: %s" % model.predict(x))
print("\n")

