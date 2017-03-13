import random as rnd
import numpy as np

# The weight is the neural value -- the network's retained "knowledge"
# weights = [ 1 * rnd.random(), 1 *rnd.random() ]
weights = [ 2000 * rnd.random(), 2000 *rnd.random() ]
# weights = [ -99, -99 ]

# An N-input neuron
def neuron(input):
    alpha = 0.1 #3

    def activate(x):
        # return math.tanh(x)
        # return (x > 0) * x
        return x

    def v_sum(v1, v2):
        assert (len(v1) == len(v2))

        out = activate(alpha * np.array(v1).dot.array(v2))

        # out2 = 0
        # for i in range(len(v1)):
        #     out2 += activate(alpha * v1[i] * v2[i])
        # assert( out == out2)
        return out

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
        ,[1.0, 1.0]
    ]

    goals = [
        #  0.0
        # ,0.0
        # ,0.0
        # ,1.0

         0.0
        ,1.0
        ,1.0
        ,0.0
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


learn( 500 )

predict( [
      [0, 0]
    , [0, 1]
    , [1, 0]
    , [1, 1]

    ])

