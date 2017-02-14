import random as rnd
import math
import numpy as np

# The weight is the neural value -- the network's retained "knowledge"
weights = [ rnd.random(), rnd.random() ]

# An N-input neuron
def neuron(input):
    alpha = 3

    # Custom relaxed rectified linear unit
    def activation(x):
        return math.tanh(x)

    def v_sum(v1, v2):
        assert (len(v1) == len(v2))
        out = 0

        out = activation(np.dot(alpha * np.array(v1), np.array(v2)))

        out2 = 0
        for i in range(len(v1)):
            out2 += activation(alpha * v1[i] * v2[i])
        # assert( out == out2)
        return out2

    out = v_sum(input, weights)
    return out

# Teach a-two-neuron layer to be a fuzzy "or" gate
def learn(learning_count):

    def ele_mul(number, vector):
        out = [0] * len(vector)

        for i in xrange(len(vector)):
            out[i] = number * vector[i]
        return out

    inputs = [
         [0.0, 0.0]
        ,[0.0, 1.0]
        ,[1.0, 0.0]
        # ,[1.0, 1.0]
        # , [0.2, 0.4]
        # , [0.8, 0.1]
        # , [0.0, 0.1]
        # , [0.001, 0.01]
        # , [0.8, 0.2]
        # , [0.2, 0.8]
        # , [0.8, 0.0]
        # , [0.1, 0.3]
        # , [0.35, 0.15]
        # , [1, 0]

    ]

    goals = [
         0.0
        ,1.0
        ,1.0
        # ,0.0
        #
        # ,0.0
        # ,1.0
        # ,0.0
        # ,0.0
        # ,1.0
        # ,1.0
        # ,1.0
        # ,0.0
        # ,0.0
        # ,0
    ]

    line_num = 0

    print("Initial weight:s " + str(weights))
    print("Number of inputs: " + str(len(inputs)))

    # Supervised learning loop -- adjusts weights to yield the goals
    for x in range(learning_count):

        for row in range(len(inputs)):
            prediction = neuron(inputs[row])

            # Learn from error amount and direction
            delta = prediction - goals[row]
            # error = delta ** 2

            gradient = ele_mul(delta, inputs[row])
            for w in range(len(weights)):
                weights[w] -= gradient[w]

            line_num += 1
            print(str(line_num) + ". Input value " + str(inputs[row]) + " predicts " + str(prediction) + ", for goal: " + str(goals[row]))
        print("")

    print("Learned weights: " + str(weights) + "\n")

# Use the two-neuron network to predict fuzzy "or" result
def predict( args ):

    print("Predict goals based on inputs distinct from training data:\n")

    line_num = 0
    for row in range(len(args)):
        prediction = neuron(args[row])
        line_num += 1
        print(str(line_num) + ". Input value# " + str(args[row]) + " predicts: " + str(prediction) + " as "+ str("True" if prediction > 0.5 else "False") )


learn( 525 )

predict( [
        #  [0.1, 0.2]
        # ,[0.2, 0.1]
        #
        # ,[0.0, 2.0]
        # ,[8.0, 0.0]
        #
        # ,[0.6, 0.3]
        # ,[0.3, 0.6]
        #
        # ,[0.7, 0.1]
        # ,[0.0, 0.2]
        #
        # ,[0.3, 0.1]
        # ,[0.1, 0.3]
        #
        # ,[0.6, 0.1]
        #
        # ,[0.0, 0.7]
        # ,[0.7, 0.0]
        #
        # ,[0.09, 0.99]

      [0, 0]
    , [0, 1]
    , [1, 0]
    # , [1, 1]

    ])

