
import numpy as np
import random as rnd
import math


weights = np.array([rnd.random() * 10, rnd.random() * 10, rnd.random() * 10])

def neural_network(input_vector, weights_vector):
    pred = input_vector.dot(weights_vector)
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

# input corresponds to every entry
# or the  first game of the season

input = np.array([toes[0],wlrec[0],nfans[0]])
pred = neural_network(input,weights)
print("Predicate: " + str(pred))
print(rnd.random())