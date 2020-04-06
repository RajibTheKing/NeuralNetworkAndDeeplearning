# modified based on http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'}]

plot_labels = ["train score", "test score"]

hidden_layer_sizes = [(50,), (100,), (200,), (50,50), (50,100), (100,50), (100,100), (100,200), (200,100)]

def plot_on_dataset(X, y, name):
    # for each dataset, plot learning for each learning strategy
    print("\n\nlearning on dataset %s\n" % name)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
    print("Number of training samples: %d" % len(X_train))
    print("Number of test samples: %d" % len(X_test))
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for ax, label, param in zip(axes, labels, params):
        train_score = []
        test_score = []
        print("training: %s" % label)
        ax.set_title(name + '\n' + label)
        ax.set_xlabel("number of hidden neurons")
        ax.set_xticks([i for i in range(len(hidden_layer_sizes))])
        ax.set_xticklabels(hidden_layer_sizes)
        for hls in range(len(hidden_layer_sizes)):
            # create multi-layer perceptron
            mlp = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes[hls],
                            verbose=0, random_state=0, max_iter=max_iter, **param)
            print("hidden layer size:")
            print( mlp.hidden_layer_sizes)
            
            # train multi-layer perceptron
            mlp.fit(X_train, y_train)
            
            # print network information
            # Number of network inputs:
            print("number of network inputs: %d" % len(X_train[0]))

            # Number of network outputs:
            print("number of network outputs: %d" % mlp.n_outputs_)

            # Number of network layers (counting LAYERS, not sets of trainable weights!):
            print("number of network layers: %d" % mlp.n_layers_)              

            print("Training set score: %f" % mlp.score(X_train, y_train))
            train_score.append(mlp.score(X_train, y_train))
            print("Training set loss: %f" % mlp.loss_)
            print("Test set score: %f\n" % mlp.score(X_test, y_test))
            test_score.append(mlp.score(X_test, y_test))
        for score, plot_label, args in zip([train_score, test_score], plot_labels, plot_args):
            ax.plot(score, label=plot_label, **args)
        ax.legend(plot_labels, loc="upper right")
    plt.show()

# load / generate some toy datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine() # NEU
diabetes = datasets.load_diabetes() # NEU
data_sets = [(iris.data, iris.target),
             (digits.data, digits.target),
             (wine.data, wine.target), # NEU
             (diabetes.data, diabetes.target), # NEU
             datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
             datasets.make_moons(noise=0.3, random_state=0)]

for data, name in zip(data_sets, ['iris', 'digits',
                                  'wine', 'diabetes', # NEU
                                  'circles', 'moons']):
    plot_on_dataset(*data, name=name)

