import random
import numpy as np
import generate_maze
from math import sqrt

# LABYRINTH = np.array([[2, 0, 0, 0, 0],
#                       [0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1],
#                       [0, 1, 0, 0, 2]])
LABYRINTH = generate_maze.get_maze()
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.6
POPULATION_SIZE = 100
ITERATIONS_COUNT = 100
EPS = 0

class GeneticLabyrinth:
    def __init__(self, labyrinth = LABYRINTH):
        self.labyrinth = labyrinth
        self.height = len(labyrinth)
        self.width = len(labyrinth[0])
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        for i in range(self.height):
            for j in range(self.width):
                if self.labyrinth[i][j] == 2:
                    if self.start_point == (-1, -1):
                        self.start_point = (i, j)
                    elif self.end_point == (-1, -1):
                        self.end_point = (i, j)
                    else:
                        raise Exception("Inappropriate labyrinth")
        if self.end_point == (-1, -1):
            raise Exception("Inappropriate labyrinth")

    def create(self):
        chromosome = np.reshape(np.random.randint(low=0, high=2, size=self.width*self.height), (self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.labyrinth[i][j] == 1:
                    chromosome[i][j] = 0
        chromosome[self.start_point[0]][self.start_point[1]] = 1
        chromosome[self.end_point[0]][self.end_point[1]] = 1
        return chromosome

    def create_population(self, population_size):
        return [self.create() for _ in range(population_size)]

    def neighbours(self, point):
        neighbours_list = []
        if point[0] > 0:
            neighbours_list.append((point[0] - 1, point[1]))
        if point[1] > 0:
            neighbours_list.append((point[0], point[1] - 1))
        if point[0] < self.height - 1:
            neighbours_list.append((point[0] + 1, point[1]))
        if point[1] < self.width - 1:
            neighbours_list.append((point[0], point[1] + 1))
        return neighbours_list

    def is_valid_path(self, chromosome):
        # wave algorithm
        path = [[-1 for i in range(self.width)] for j in range(self.height)]
        path[self.start_point[0]][self.start_point[1]] = 0
        opened = [self.start_point]
        for p in opened:
            for neighbour in self.neighbours(p):
                if path[neighbour[0]][neighbour[1]] == -1 and chromosome[neighbour[0]][neighbour[1]] == 1:
                    path[neighbour[0]][neighbour[1]] = path[p[0]][p[1]] + 1
                    opened.append(neighbour)
                if neighbour == self.end_point:
                    return True
        return False

    # def eval(self, chromosome):
    #     score = 0
    #     penalty = 1000 * self.height * self.width
    #     if chromosome[self.start_point[0]][self.start_point[1]] == 0:
    #         score += penalty
    #     if chromosome[self.end_point[0]][self.end_point[1]] == 0:
    #         score += penalty
    #
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             if self.labyrinth[i][j] == 1 and chromosome[i][j] == 1:
    #                 score += penalty
    #             score += chromosome[i][j]
    #
    #     if not self.is_valid_path(chromosome):
    #         score += 1000*penalty
    #
    #     return score
    def eval(self, chromosome):
        score = 0
        # euclid distance
     #   score = sqrt((self.end_point[0] - self.width) ** 2 + (self.end_point[1] - self.height) ** 2)
        # manhattan distance
        score = int(abs(abs(self.end_point[0]) - abs(self.width)) + abs(abs(self.end_point[1]) - abs(self.height)))
        penalty = 1000 * self.height * self.width
        if chromosome[self.start_point[0]][self.start_point[1]] == 0:
            score += penalty
        if chromosome[self.end_point[0]][self.end_point[1]] == 0:
            score += penalty

        for i in range(self.height):
            for j in range(self.width):
                if self.labyrinth[i][j] == 1 and chromosome[i][j] == 1:
                    score += penalty
                score += chromosome[i][j]

        if not self.is_valid_path(chromosome):
            score += 1000*penalty

        return score

    def select(self, chromosomes, number_to_select):
        values = [self.eval(chromosome) for chromosome in chromosomes]
        max_value = max(values)
        min_value = min(values)
        order = np.argsort(values)
        values_ordered = np.array(values)[order]
        chromosomes_ordered = np.array(chromosomes)[order]
        # values_for_prob = [max_value - value for value in values_ordered]
        # values_for_prob = [int(1000*np.exp(-i*0.05)) for i in range(len(chromosomes_ordered))]
        values_for_prob = [int((max_value - value)*np.exp(-i*0.005)) for i, value in enumerate(values_ordered)]
        # print(values_for_prob)
        values_for_prob = np.cumsum(np.array(values_for_prob))
        selected = []
        selected_values = []
        for i in range(number_to_select):
            gen = random.randint(0, np.max(values_for_prob))
            selected_index = np.argmax(values_for_prob >= gen)
            selected.append(chromosomes_ordered[selected_index])
            selected_values.append(values_ordered[selected_index])
        # print(selected_values)
        return selected

    def best(self, chromosomes):
        values = [self.eval(chromosome) for chromosome in chromosomes]
        return chromosomes[np.argmin(values)]

    def mutate(self, chromosome):
        count_mutations = int(self.width * self.height * MUTATION_PROB)
        res_chromosome = np.copy(chromosome)
        for mut in range(count_mutations):
            i, j = self.start_point
            while (i, j) == self.start_point or (i, j) == self.end_point:
                i = random.randint(0, self.height - 1)
                j = random.randint(0, self.width - 1)
            if res_chromosome[i][j] == 0:
                res_chromosome[i][j] = 1
            else:
                res_chromosome[i][j] = 0
        return res_chromosome

    def mutate_population(self, chromosomes):
        chromosomes_to_mutate = self.select(chromosomes, int(MUTATION_PROB * POPULATION_SIZE))
        return [self.mutate(chromosome) for chromosome in chromosomes_to_mutate]

    def crossover(self, chromosome1: np.array, chromosome2: np.array):
        y1_prev = random.randint(0, self.height - 1)
        y2_prev = random.randint(0, self.height - 1)
        y1 = min(y1_prev, y2_prev)
        y2 = max(y1_prev, y2_prev)
        x1_prev = random.randint(0, self.width - 1)
        x2_prev = random.randint(0, self.width - 1)
        x1 = min(x1_prev, x2_prev)
        x2 = max(x1_prev, x2_prev)
        chromosome1_new = np.copy(chromosome1)
        chromosome2_new = np.copy(chromosome2)
        chromosome1_new[y1:y2, x1:x2] = chromosome2[y1:y2, x1:x2]
        chromosome2_new[y1:y2, x1:x2] = chromosome1[y1:y2, x1:x2]
        return chromosome1_new, chromosome2_new

    def crossover_population(self, chromosomes):
        chromosomes_to_crossover = self.select(chromosomes, int(CROSSOVER_PROB * POPULATION_SIZE))
        result = []
        for i in range(len(chromosomes_to_crossover)//2):
            chr1, chr2 = self.crossover(chromosomes_to_crossover[2*i], chromosomes_to_crossover[2*i + 1])
            result.append(chr1)
            result.append(chr2)
        return result

    def main(self, pop_size = POPULATION_SIZE):
        chromosomes = self.create_population(pop_size)
        best = self.best(chromosomes)
        for iteration in range(ITERATIONS_COUNT):
            print(f"{iteration}: {self.eval(best)}")
            mutated = self.mutate_population(chromosomes)
            crossovered = self.crossover_population(chromosomes)
            selected = self.select(chromosomes, POPULATION_SIZE - len(mutated) - len(crossovered))
            chromosomes = mutated + crossovered + selected
            best = self.best(chromosomes)
        print(f"result: ")
        print(self.best(chromosomes))
        print(f"the value is {self.eval(best)}")

if __name__ == '__main__':
    gen_lab = GeneticLabyrinth()
    gen_lab.main()
    # chromosomes = gen_lab.create_population(10)
    # selected = gen_lab.select(chromosomes, 3)