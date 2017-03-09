import random

# This weight is the neural value; the "knowledge"
# At first we know nuthin'
weight = -random.random() * 2
initial_weight = weight

# We seek a neuron that predicts a value 0f 0.8 for an input of 0.5
goal = 0.8
# This the hello-world demo. Most neurons  take an array of inputs instead of a scalar.
input_value = 0.5

fuzz = 1e-25

# Input-to-output result non-linear if neuron were to cascade to a hidden laryer
def activate(x):
    return max(0, x)

# This is the actual single-input neuron.
def neuron(input):
    return activate(weight * input)

# "Supervised" learning:
counter = 0
circuit_breaker = int(1e5)

for x in range(0, circuit_breaker):
    prediction = neuron(input_value)

    delta = prediction - goal
    error = delta ** 2

    # Adjust weight from error amount and sign of error
    weight -= delta * input_value

    counter += 1
    print(str(counter) + ". Prediction: " + str(prediction) + ", goal: " + str(goal))

    circuit_breaker = circuit_breaker -1
    if error < fuzz:
        break


print("Initial weight: " + str(initial_weight))
print("Learned weight: " + str(weight))