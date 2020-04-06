# modified based on http://gonzalopla.com/deep-learning-nonlinear-regression/
# Numeric Python Library.
import numpy as np
# Python Data Analysis Library.
import pandas
# Scikit-learn Machine Learning Python Library modules.
# multi-layer perceptron regressor
from sklearn.neural_network import MLPRegressor
#   Preprocessing utilities.
from sklearn import preprocessing
#   Cross-validation utilities.
#depricated .... from sklearn import cross_validation
from sklearn.model_selection import train_test_split
# Python graphical library
import matplotlib.pyplot as plt

# Peraring dataset
# Imports csv into pandas DataFrame object.
Eckerle4_df = pandas.read_csv("Eckerle4.csv", header=0)
 
# Converts dataframes into numpy objects.
Eckerle4_dataset = Eckerle4_df.values.astype("float32")
# Slicing all rows, second column...
X = Eckerle4_dataset[:,1]
# Slicing all rows, first column...
y = Eckerle4_dataset[:,0]
 
# plot data
plt.plot(X,y, color='red')
plt.legend(labels=["data"], loc="upper right")
plt.title("data")
plt.show()

# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = ( X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)).reshape(-1) )
 
# Preparing test and train data: 60% training, 40% testing.
X_train, X_test, y_train, y_test = train_test_split( \
    X_scaled, y_scaled, test_size=0.40, random_state=3)

# hidden layer sizes
hidden_layer_size = (128, 64, 64) # FIX!!!

# construct MLP regressor

mlp = MLPRegressor( hidden_layer_sizes=hidden_layer_size,
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=256,
                    learning_rate_init=0.001,
                    alpha=0.002,
                    batch_size=2)
# train MLP
mlp.fit(X_train, y_train)

plt.plot(mlp.loss_curve_, color='red')
plt.legend(labels=["loss"], loc="upper right")
plt.title("training loss")
plt.show()

# print network information
# Number of network inputs:
print("number of network inputs: %d" % len(X_train[0]))

# Number of network outputs:
print("number of network outputs: %d" % mlp.n_outputs_)

# Number of network layers (counting LAYERS, not sets of trainable weights!):
print("number of network layers: %d" % mlp.n_layers_)              

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Training set loss: %f" % mlp.loss_)
print("Test set score: %f\n" % mlp.score(X_test, y_test))

# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
predicted = mlp.predict(X_test)
plt.plot(y_scaler.inverse_transform(predicted.reshape(-1,1)), color="blue")
plt.plot(y_scaler.inverse_transform(y_test.reshape(-1,1)), color="green")
plt.legend(labels=["predicted", "target"], loc="upper right")
plt.title("evaluation on test corpus")
plt.show()
