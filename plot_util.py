import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_function(values):
    plt.title("Fitness function")
    plt.xlabel("Iteration")
    plt.ylabel("log2(log2(value))")
    plt.plot([i for i in range(len(values))], np.log2(np.log2(values)), color="red")
    plt.show()
