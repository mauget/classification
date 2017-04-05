
import numpy as np

learniing_cycles = 300  # Number of "lessons"
alpha = 0.5             # Like a step size. Empirically-set dampening for error correction
bias = -0.001

np.random.seed(1)

# Sigmoid function applied to all items of x numpy array
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Slope of sigmoid  applied to all items of x  numpy array
# Param is a nummpy array
def sigmoidDeriv(x):
    sm = sigmoid(x)
    return sm * (1.0 - sm)

def tanhDeriv(x):
    return 1.0 - tanh(x) ** 2

def tanh(x):
    return np.tanh(x)

def reluDeriv(x):
    return (x > 0) * 1

def relu(x):
    return (x > 0) * x

def heavisideDeriv(x):
    return (abs(x) < 0.1) * 1

def heaviside(x):
    return (abs(x) >= 0 ) * 1

def activation(x):
    # return heaviside(x)
    # return relu(x)
    # return sigmoid(x)
    return tanh(x)

def activationDeriv(x):
    # return heavisideDeriv(x)
    # return reluDeriv(x)
    # return sigmoidDeriv(x)
    return tanhDeriv(x)

# Param is an int
def shouldLogNow(line_no):
    return line_no % 10 == 0

# Input the following. We'll learn to  ....
inputs = np.array(
    [[ 0, 0 ],
     [ 0, 1 ],
     [ 1, 0 ],
     [ 1, 1 ]
    ] )

# ... predict these target fuzzy XOR meanings, row-for-row. Values > 0.5  mean true
goals = np.array(
    # XOR
    [[0],
     [1],
     [1],
     [0]
    ] )

    # AND
    # [[0],
    #  [0],
    #  [0],
    #  [1]
    # ] )

    # OR
    # [[0],
    #  [1],
    #  [1],
    #  [1]
    # ] )

hidden_size = inputs.shape[0]  # 4 rows of input. i.e. the number of  neurons in the single hidden layer
input_size  = inputs.shape[1]  # 2 cols of input. i.e. the number of items in an input row

weights_0_1 = 20 * np.random.random((input_size, hidden_size)) -1
weights_1_2 = 20 * np.random.random((hidden_size,1)) - 1

def logKnowledge():
    np.set_printoptions(threshold='nan')
    print( "\nInput weights: " )
    print(  weights_0_1.T)
    print( "\nHidden weights: " )
    print(  weights_1_2)
    print( "\nPredictions: ")
    print(goals)
    print("")

print("\nRandomized input weights follow ...")
logKnowledge()

line_num = 0

for iteration in xrange(learniing_cycles):
    layer_2_error = 0

    # Logging
    line_num += 1
    if shouldLogNow(line_num):
        print(str(line_num))

    for i in xrange(len(inputs)):

        # The sigmoid function makes the model non-linear.
        # Otherwise the multiplications of succesive layers could be carried out in one layer ... to no advantage.
        layer_0 = inputs[i : i + 1]
        layer_1 = activation(np.dot(layer_0, weights_0_1) + bias)
        layer_2 = np.dot(layer_1, weights_1_2)

        # layer_2 is the prediction layer.
        # Teach it incrementally by correcting weight values and directions
        # # Apply small corrections minimize difference between the result and the known prediction value.
        # numpy is our friend hete by working on entire rows using efficient code.
        # Note the sigmoid derivative function that selectively attenuates back-layer inputs that had little or no
        # contribution.

        # Error diffs -- back-propagate to layer_1_delta
        layer_2_delta = layer_2 - goals[i]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * activationDeriv(layer_1)

        # Forward-propagate lowest error
        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

        # Logging
        if shouldLogNow(line_num):
            print( "\t-" + str(i)+  ". Output: " + str(layer_2[0]) + " => " + str(layer_2[0] > 0.5 )  )

logKnowledge()