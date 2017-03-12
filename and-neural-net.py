
import numpy as np

learniing_cycles = 200  # Number of "lessons"
alpha = 0.5             # Like a step size. Empirically-set dampening for error correction

np.random.seed(1)

# Sigmoid function applied to all items of x numpy array
def relu(x):
    return (x > 0) * x

# Param is an int
def shouldLogNow(line_no):
    return line_no % 10 == 0

# Sigmoid function applied to all items of x numpy array
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Input the following. We'll learn to  ....
inputs = np.array(
    [[ 0, 0 ],
     [ 0, 1 ],
     [ 1, 0 ],
     [ 1, 1 ]
    ] )

# ... predict these target fuzzy XOR meanings, row-for-row. Values > 0.5  mean true
predictions = np.array(
    [[0],
     [0],
     [0],
     [1]
    ])


weights_0_1 = 20 * np.random.random((inputs.shape[1], inputs.shape[0]) ) - 1

def logKnowledge():
    np.set_printoptions(threshold='nan')
    print( "\nInput weights: " )
    print(  weights_0_1.T)
    print( "\nPredictions: ")
    print( predictions)
    print("")

print("\nRandomized input weights follow ...")
logKnowledge()

line_num = 0

for iteration in xrange(learniing_cycles):

    # Logging
    line_num += 1
    if shouldLogNow(line_num):
        print(str(line_num))

    for i in xrange(len(inputs)):
        layer_0 = inputs[i : i + 1]
        layer_1 = (layer_0.dot(weights_0_1))

        layer_1_delta = layer_1 - predictions[i]

        # propagate lowest error
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)

        # Logging
        if shouldLogNow(line_num):
            print( "\t-" + str(i)+  ". Output: " + str(layer_1.T[i]) + " => " + str(layer_1.T[i] >= 0.5 )  )

logKnowledge()