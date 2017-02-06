import random
import math
import numpy as np

# The weight is the neural value -- the "knowledge"
weight = random.random() * 10

initial_weight = weight
counter = 0

# We want a neuron to predict a value 0f 0.8 for an input of 0.5
goal = 0.8
input_value = 0.5

fuzz = 1e-25 #1e-10

# # Sigmoid
# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))

# Relaxed rectified linear unit
def relu(x):
    return 0 if x <= 0 else x

# This is a single-input neuron!
def neuron(input):
    alpha = 3

    # prediction = sigmoid(alpha * weight * input)
    # prediction = math.tanh(alpha * weight * input)
    prediction = relu(alpha * weight * input)
    # prediction = relu(alpha * np.array([input]).dot(np.array([weight])))

    return prediction

# "Supervised" learning:
circuit_breaker = int(1e5)
is_learning = True
for x in range(0, circuit_breaker):
    prediction = neuron(input_value)

    # Learn from error amount and direction here
    delta = prediction - goal
    error = delta ** 2

    gradient = delta * input_value
    weight -= gradient

    counter += 1
    print(str(counter) + ". Prediction: " + str(prediction) + ", goal: " + str(goal))

    circuit_breaker = circuit_breaker -1
    if error < fuzz:
        break



print("Initial weight: " + str(initial_weight))
print("Learned weight: " + str(weight))
# print("Iterations: " + str(iterations));