import random as rnd
import sys
import math
import numpy as np

# The weight is the neural value -- the "knowledge"
weights =  [rnd.random(), rnd.random()]

print("Initial weight:s " + str(weights))

inputs = [
     [0.0, 0.0]
    ,[0.0, 1.0]
    ,[1.0, 0.0]
    # ,[1.0, 1.0]

    , [0.2, 0.4]
    , [0.8, 0.1]
    , [0.0, 0.1]
    , [0.001, 0.01]
    , [0.8, 0.2]
    , [0.8, 0.0]
]

print("Number of inputs: " + str(len(inputs)))

goals = [
     0.0
    ,1.0
    ,1.0
    # ,0.0
    ,0.0
    ,1.0
    ,0.0
    ,0.0
    ,1.0
    ,1.0
]

line_num = 0

alpha = 1

# Relaxed rectified linear unit
def relu(x):
    return 0 if x <= 0 else x

def v_sum(v1, v2):
    assert(len(v1) == len(v2))
    out = 0

    for i in range(len(v1)):
        out += relu( alpha * v1[i] * v2[i] )
    return out

def ele_mul(number, vector):
    out = [0] * len(vector)

    for i in xrange(len(vector)):
        out[i] = number * vector[i]
    return out

# An N-input neuron
def neuron(input, weights):
    out = v_sum( input, weights )
    return out

# "Supervised" learning:
learning_cycles = 10000

for x in range(learning_cycles):

    for row in range(len(inputs)):
        prediction = neuron(inputs[row], weights)

        # Learn from error amount and direction
        delta = prediction - goals[row]
        # error = delta ** 2

        gradient = ele_mul(delta, inputs[row])
        for w in range(len(weights)):
            weights[w] -= gradient[w]

        line_num += 1
        print(str(line_num) + ". Predicted input value " + str(inputs[row]) + ": " + str(prediction) + ", goal: " + str(goals[row]))
    print("")

print("Learned weights: " + str(weights))

# Use the two-neuron network


args = [
     [0.0, 0.0]
    ,[0.0, 1.0]
    ,[1.0, 0.0]

    ,[0.2, 0.4]
    ,[0.8, 0.1]
    ,[0.0, 0.1]
    ,[0.001, 0.01]
    ,[0.8, 0.2]
    ,[0.8, 0.0]
]

line_num = 0
for row in range(len(args)):
    prediction = neuron(args[row], weights)
    line_num += 1
    print(str(line_num) + ". Predicted input value# " + str(args[row]) + ": " + str(prediction) )

