import random

# This weight is the neural value -- the "knowledge"
# At first we're Sgt Shultz: we know nuthing!
weight = 300 * random.random()
initial_weight = weight

# We seek a neuron that predicts a value 0f 0.8 for an input of 0.5
goal = 0.8

# It's a demo! Most neurons take an input array instead of a scalar.
input_value = 0.5

diff_fuzz = 1e-25

# This is the actual single-input neuron.
def neuron(input):
    return weight * input

# "Supervised" learning:
counter = 0
circuit_breaker = int(1e5)

for x in range(0, circuit_breaker):
    prediction = neuron(input_value)

    # Adjust weight from delta slope magnitube and sign
    delta = prediction - goal
    weight -= delta * input_value

    counter += 1
    print(str(counter) + ". Prediction " + str(prediction) + " via weight " + str(weight) + ". Seeking goal " + str(goal))

    circuit_breaker -= 1
    error = delta ** 2
    if error < diff_fuzz:
        break


print("Initial weight: " + str(initial_weight))
print("Learned weight: " + str(weight))