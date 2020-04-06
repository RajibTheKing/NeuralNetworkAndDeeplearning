"""
min-char-rnn.py: Written by Andrej Karpathy, minimally modified by CM
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
from https://gist.github.com/karpathy/d4dee566867f8291f086
adjusted for python 3 (CM, put brackets to print command)
BSD License
"""
import numpy as np
import time

# data I/O
data = open('input_short.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25   # number of steps to unroll the RNN for, must be smaller than input text!
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias


def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)): # remark: xrange -> range
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss) (negative log probability of the correct answer)
    # remark CM: this is actually the log-likelihood loss; ps[t][targets[t],0] extracts the output probability ps[t] at the component of the target (targets[t],0); second index 0 because it's a column vector with 1 column  
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))): # remark: xrange -> range
  # remark CM: going backwards the seq_length examples
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    # remark CM: As explained on website, by computing the gradient, the score of the correct class is subtracted by 1 
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients	
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n): # remark: xrange -> range
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

# levenshtein distance for comparison of text strings
# from http://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])

# main program
global_start_time = time.time()    
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
#while True: # original code
while n < 10000:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  # remark CM: sampling because the RNN outputs PROBABILITIES for each character at each time step
  #            the actual letter is sampled from these probabilities
  if n % 100 == 0:
    # beginning at the first letter of the current sequence, sample 200 characters
    sample_ix = sample(hprev, inputs[0], 200) 
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\nsample 200 letters from current position:\n%s \n----' % (txt, ))

    # alternatively, starting with the first letter of the text, sample 100 characters
    sample_ix = sample(np.zeros_like(hprev), char_to_ix[data[0]], 100) 
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('sample 100 letters from beginning of text:\n%s \n----' % (data[0] + txt) ) # data[0] added because first letter is input and not predicted
    print()
    
  # forward seq_length characters through the net and fetch gradient
  # remark CM: hprev keeps track of the hidden state vector at the end of the current batch
  #            which is fed in to the forward propagation of the next batch; i.e. it allows 
  #            to correctly initialise the subsequent batch in the next forward iteration
  #            so the hidden state vectors are correctly propagated from batch to batch
  #            (however, backpropagation is performed only for the last seq_length steps)
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  
training_time = time.time() - global_start_time
print('Training duration (s) : ', training_time)
  
# final test: starting with the first letter of the text, sample as many characters as there were in the training file
# NOTE: this is like a test on the training data, to check how good the text was remembered

print("\nFinal test: Trying to reproduce training sequence; predicted text:\n")
# alternatively, starting with the first letter of the text, sample as many characters as there are in the training text
sample_ix = sample(np.zeros_like(hprev), char_to_ix[data[0]], len(data)) 
predicted = ''.join(ix_to_char[ix] for ix in sample_ix)
predicted = data[0] + predicted # must insert first character, since it is not predicted
print("%s\n" % predicted)
num_errors = levenshtein(predicted, data)
print("\n... there are %d errors" % num_errors)
  