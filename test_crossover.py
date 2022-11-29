import random
import numpy as np
import generate_maze
from plot_util import plot_fitness_function

LABYRINTH = np.array([[1, 2, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1],
                      [1, 0, 0, 0, 1, 1, 1],
                      [1, 0, 1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 2, 1]])
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.6
POPULATION_SIZE = 100
ITERATIONS_COUNT = 100
EPS = 0
CROSSOVER_TYPE = 'standart_with_probability'

class GeneticLabyrinth:
    def __init__(self, labyrinth = LABYRINTH, crossover = CROSSOVER_TYPE):
        self.labyrinth = labyrinth
        self.crossover_type = crossover
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

    def eval(self, chromosome):
        score = 0
        penalty = self.height * self.width
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
            score += np.sqrt(penalty)*penalty

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

    def crossover(self, chromosome1: np.array, chromosome2: np.array, type = 'standart'):
        y1_prev = random.randint(0, self.height - 1)
        y2_prev = random.randint(0, self.height - 1)
        y1 = min(y1_prev, y2_prev)
        y2 = max(y1_prev, y2_prev)
        x1_prev = random.randint(0, self.width - 1)
        x2_prev = random.randint(0, self.width - 1)
        x1 = min(x1_prev, x2_prev)
        x2 = max(x1_prev, x2_prev)
        if self.crossover_type == 'standart':
            chromosome1_new = np.copy(chromosome1)
            chromosome2_new = np.copy(chromosome2)
            chromosome1_new[y1:y2, x1:x2] = chromosome2[y1:y2, x1:x2]
            chromosome2_new[y1:y2, x1:x2] = chromosome1[y1:y2, x1:x2]
            return chromosome1_new, chromosome2_new
        elif self.crossover_type == 'standart_with_probability':
            best_chromosome = (np.copy(chromosome1) if self.eval(chromosome1) < self.eval(chromosome2) else np.copy(chromosome2))
            worst_chromosome = (np.copy(chromosome1) if self.eval(chromosome1) > self.eval(chromosome2) else np.copy(chromosome2))
            if random.random() >= 0.05:
                chromosome1_new = np.copy(worst_chromosome)
             #   chromosome2_new = np.copy(chromosome2)
                chromosome1_new[y1:y2, x1:x2] = best_chromosome[y1:y2, x1:x2]
              #  chromosome2_new[y1:y2, x1:x2] = chromosome1[y1:y2, x1:x2]
            else:
                chromosome1_new = np.copy(best_chromosome)
                #   chromosome2_new = np.copy(chromosome2)
                chromosome1_new[y1:y2, x1:x2] = worst_chromosome[y1:y2, x1:x2]
            return chromosome1_new
        elif self.crossover_type == 'get_best_cells':
            best_chromosome = (np.copy(chromosome1) if self.eval(chromosome1) < self.eval(chromosome2) else np.copy(chromosome2))
            worst_chromosome = (np.copy(chromosome1) if self.eval(chromosome1) > self.eval(chromosome2) else np.copy(chromosome2))
            chromosome1_new = np.copy(worst_chromosome)
            for i in range(self.height):
                for j in range(self.width):
                    if random.random() < 0.4:
                        chromosome1_new[i][j] = worst_chromosome[i][j]
            return chromosome1_new



    def crossover_population(self, chromosomes):
        chromosomes_to_crossover = self.select(chromosomes, int(CROSSOVER_PROB * POPULATION_SIZE))
        result = []
        for i in range(len(chromosomes_to_crossover)//2):
            if self.crossover_type == 'standart':
                chr1, chr2 = self.crossover(chromosomes_to_crossover[2*i], chromosomes_to_crossover[2*i + 1])
                result.append(chr1)
                result.append(chr2)
            elif self.crossover_type == 'standart_with_probability' or self.crossover_type == 'get_best_cells':
               # chr1, chr2 = self.crossover(chromosomes_to_crossover[2*i], chromosomes_to_crossover[2*i + 1])
                chr1 = self.crossover(chromosomes_to_crossover[2*i], chromosomes_to_crossover[2*i + 1])
                result.append(chr1)
               # result.append(chr2)
        return result

    def main(self, pop_size = POPULATION_SIZE):
        chromosomes = self.create_population(pop_size)
        best = self.best(chromosomes)
        fitnessValues = np.zeros(ITERATIONS_COUNT)
        for iteration in range(ITERATIONS_COUNT):
            bestFitnessValue = self.eval(best)
            # print(f"{iteration}: {bestFitnessValue}")
            fitnessValues[iteration] = bestFitnessValue
            mutated = self.mutate_population(chromosomes)
            crossovered = self.crossover_population(chromosomes)
            selected = self.select(chromosomes, POPULATION_SIZE - len(mutated) - len(crossovered))
            chromosomes = mutated + crossovered + selected
            best = self.best(chromosomes)
      #  print(f"result: ")
        print(self.eval(best))
     #   print(f"the value is {self.eval(best)}")
    #    plot_fitness_function(fitnessValues)
        return self.eval(best)

if __name__ == '__main__':
    num_to_compare = 20
    great_res_bound = 40

    average_prob = 0
    great_res_prob = 0
    for i in range(num_to_compare):
        labyrinth = generate_maze.get_maze()
        gen_lab = GeneticLabyrinth(labyrinth)
        res = gen_lab.main()
        average_prob += res
        if res < great_res_bound:
            great_res_prob += 1
       # print(res)

    average = 0
    great_res_standard = 0
    for i in range(num_to_compare):
        labyrinth = generate_maze.get_maze()
        gen_lab = GeneticLabyrinth(labyrinth, crossover='standart')
        res = gen_lab.main()
        average += res
        if res < great_res_bound:
            great_res_standard += 1
       # print(res)

    average_best_cells = 0
    great_res_best_cells = 0
    for i in range(num_to_compare):
        labyrinth = generate_maze.get_maze()
        gen_lab = GeneticLabyrinth(labyrinth, crossover='get_best_cells')
        res = gen_lab.main()
        average_best_cells += res
        if res < great_res_bound:
            great_res_best_cells += 1
    # print(res)

    print('standard_with_probability')
    print('average fitness ', average_prob/num_to_compare)
    print("good percentage ", great_res_prob/num_to_compare)
    print('standard')
    print('average fitness ', average/num_to_compare)
    print("good percentage ", great_res_standard/num_to_compare)
    print('get_best_cells')
    print('average fitness ', average_best_cells/num_to_compare)
    print("good percentage ", great_res_best_cells/num_to_compare)