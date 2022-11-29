import matplotlib.pyplot as plt
import numpy as np

EPS = 0.005

def plot_fitness_function(*args: np.array):
    plt.title("Fitness function")
    plt.xlabel("Iteration")
    plt.ylabel("log2(log2(value))")

    for j, values in enumerate(args):
        plt.plot([i for i in range(len(values))], np.log2(np.log2(values)) - EPS*j)
    plt.show()

def plot_fitness_function_unlogged(*args: np.array):
    plt.title("Fitness function")
    plt.xlabel("Iteration")
    plt.ylabel("value")
    for j, values in enumerate(args):
        plt.plot([i for i in range(len(values))], values - EPS*j)
    plt.show()