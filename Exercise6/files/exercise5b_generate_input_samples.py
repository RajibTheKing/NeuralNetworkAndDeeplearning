import numpy as np
import numpy.random as rand

###-------------------------------------
# artificially generate training samples
###-------------------------------------

# inputs
dim = 2                # dimension of input vector
numSamples = 100       # number of training samples per class
numCenterInArray = 1   # number of centers used in rand function below
center1 = np.array([[0.2, 0.2]]) # center of class 1
center2 = np.array([[0.4, 0.4]]) # center of class 2
center3 = np.array([[0.7, 0.3]]) # center of class 3
center4 = np.array([[0.2, 0.5]]) # center of class 4
rand_norm1 = 0.05 * rand.randn(numSamples, numCenterInArray, dim) # random values added to center 1
rand_norm2 = 0.05 * rand.randn(numSamples, numCenterInArray, dim) # random values added to center 2
rand_norm3 = 0.05 * rand.randn(numSamples, numCenterInArray, dim) # random values added to center 3
rand_norm4 = 0.05 * rand.randn(numSamples, numCenterInArray, dim) # random values added to center 4
class1 = np.array([center1 + r for r in rand_norm1]) # generate samples of class 1
class1.shape = (numSamples*numCenterInArray, dim)           # rearrange class 1 array
class2 = np.array([center2 + r for r in rand_norm2]) # generate samples of class 2
class2.shape = (numSamples*numCenterInArray, dim)           # rearrange class 2 array 
class3 = np.array([center3 + r for r in rand_norm3]) # generate samples of class 3
class3.shape = (numSamples*numCenterInArray, dim)           # rearrange class 3 array
class4 = np.array([center4 + r for r in rand_norm4]) # generate samples of class 4
class4.shape = (numSamples*numCenterInArray, dim)           # rearrange class 4 array
   
tmp = np.append(class1, class2, 0)
tmp1 = np.append(tmp, class3, 0)
inputOrdered = np.append(tmp1, class4, 0) # input set ORDERED 
input = inputOrdered
rand.shuffle( input )

np.savetxt('exercise3a_input.txt', input, fmt='%f')
np.savetxt('exercise3a_class1.txt', class1, fmt='%f')
np.savetxt('exercise3a_class2.txt', class2, fmt='%f')
np.savetxt('exercise3a_class3.txt', class3, fmt='%f')
np.savetxt('exercise3a_class4.txt', class4, fmt='%f')
np.savetxt('exercise3a_center1.txt', center1, fmt='%f')
np.savetxt('exercise3a_center2.txt', center2, fmt='%f')
np.savetxt('exercise3a_center3.txt', center3, fmt='%f')
np.savetxt('exercise3a_center4.txt', center4, fmt='%f')

#np.savetxt('exercise3a_input_ordered.txt', inputOrdered, fmt='%f')
