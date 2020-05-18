"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys
import time
import collections

myStudentNum = 183682
# Replace 12345 with your student number
random.seed(myStudentNum)


class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, random_heuristic, random_sus, uniform_pm,
                 recip_inv):
        """
        Parameters and general variables
        """

        self.population = []
        self.matingPool = []
        self.new_matingPool = []
        self._0_random_1_heuristic = random_heuristic
        self._0_random_1_sus = random_sus
        self._0_uniform_1_pmx = uniform_pm
        self._0_recip_1_inv = recip_inv
        self.best = None
        self.popSize = _popSize
        self.genSize = None
        self.mutationRate = _mutationRate
        self.maxIterations = _maxIterations
        self.iteration = 0
        self.fName = _fName
        self.data = {}

        self.readInstance()
        self.initPopulation()

    def readInstance(self):
        """
        Reading an instance from fName
        """

        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data, self._0_random_1_heuristic)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize - 1)]
        indB = self.matingPool[random.randint(0, self.popSize - 1)]

        return [indA, indB]

    def sum_total_fitness(self):
        sum_fitness = 0.0
        for ind_i in self.population:
            sum_fitness = sum_fitness + ind_i.getFitness()
        return sum_fitness

    def stochasticUniversalSampling(self):
        """
               Your stochastic universal sampling Selection Implementation
               """
        dict_pop_fitness = {}
        sum_fitness = self.sum_total_fitness()
        end_point = 0

        for ind_i in self.population:
            start_point = end_point
            numerator = ind_i.getFitness() / sum_fitness
            end_point = end_point + numerator
            dict_pop_fitness[ind_i] = [start_point, end_point]

        num_parent = self.popSize
        successive_point = end_point / num_parent

        rand_num = random.uniform(0, successive_point)

        list_val = [rand_num + i * successive_point for i in range(num_parent)]

        counter = 0

        for ind in self.population:
            item = dict_pop_fitness[ind]
            if item[0] < list_val[counter] <= item[1]:
                self.new_matingPool.append(ind)
                counter = counter + 1

        indA = self.new_matingPool[random.randint(0, len(self.new_matingPool) - 1)]
        indB = self.new_matingPool[random.randint(0, len(self.new_matingPool) - 1)]

        return indA, indB

    def find_missing(self, lst):
        return sorted(set(range(0, self.genSize)) - set(lst))

    def uniformCrossover(self, indA, indB):

        """Your Uniform Crossover Implementation  """
        cityAList = indA.genes
        cityBList = indB.genes
        childA = [0] * self.genSize

        rand_pos_list = sorted(random.sample(range(1, self.genSize), int(self.genSize / 2)))
        remain_index = self.find_missing(rand_pos_list)

        for x in range(0, len(rand_pos_list)):
            childA[rand_pos_list[x]] = cityAList[rand_pos_list[x]]

        counter = 0
        for index in range(0, len(childA)):
            if not cityBList[index] in childA:
                childA[remain_index[counter]] = cityBList[index]
                counter = counter + 1

        indA.genes = childA
        return indA

    def pmxCrossover(self, indA, indB):

        """ Your PMX Crossover Implementation
               """
        cityAList = indA.genes
        cityBList = indB.genes
        childA = [0] * self.genSize
        childB = [0] * self.genSize

        crosspoint1 = random.randint(0, self.genSize - 1)
        crosspoint2 = random.randint(0, self.genSize - 1)

        rand_pos_list = sorted([crosspoint1, crosspoint2])

        childA[rand_pos_list[0]:rand_pos_list[1] + 1] = cityBList[rand_pos_list[0]:rand_pos_list[1] + 1]
        childB[rand_pos_list[0]:rand_pos_list[1] + 1] = cityAList[rand_pos_list[0]:rand_pos_list[1] + 1]

        for index in range(0, self.genSize):
            if childA[index] != 0:
                pass
            else:
                if not cityAList[index] in childA:
                    childA[index] = cityAList[index]
                else:
                    m1 = cityAList[index]
                    while m1 in childA:
                        n1 = m1
                        m1 = childB[childA.index(m1)]
                    childA[index] = m1

        indA.genes = childA

        return indA

    def reciprocalExchangeMutation(self, ind):
        """
             Your Reciprocal Exchange Mutation implementation
               """
        if random.random() > self.mutationRate:
            return ind
        cityAList = ind.genes
        crosspoint1 = random.randint(0, self.genSize - 1)
        crosspoint2 = random.randint(0, self.genSize - 1)

        crosspoint = sorted([crosspoint1, crosspoint2])
        temp = cityAList[crosspoint[0]]
        cityAList[crosspoint[0]] = cityAList[crosspoint[1]]
        cityAList[crosspoint[1]] = temp

        ind.genes = cityAList

        ind.computeFitness()
        self.updateBest(ind)

        return ind

    def inversionMutation(self, ind):
        """
                Your Inversion Mutation implementation
                """

        if random.random() > self.mutationRate:
            return ind
        cityAList = ind.genes
        crosspoint1 = random.randint(0, self.genSize - 1)
        crosspoint2 = random.randint(0, self.genSize - 1)

        crosspoint = sorted([crosspoint1, crosspoint2])

        cityAList[crosspoint[0]:crosspoint[1]] = cityAList[crosspoint[0]:crosspoint[1]][::-1]

        ind.genes = cityAList

        ind.computeFitness()
        self.updateBest(ind)

        return ind

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        for i in range(0, self.genSize):
            if min(indexA, indexB) <= i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append(ind_i.copy())

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """

        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            if (self._0_random_1_sus == 0):
                parents = self.randomSelection()
            else:
                parents = self.stochasticUniversalSampling()

            if (self._0_uniform_1_pmx == 0):
                child = self.uniformCrossover(parents[0], parents[1])
            else:
                child = self.pmxCrossover(parents[0], parents[1])

            if (self._0_recip_1_inv == 0):
                mutated_child = self.reciprocalExchangeMutation(child)
            else:
                mutated_child = self.inversionMutation(child)

                self.population[i] = mutated_child

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        # self.new_matingPool[:] = []
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print("Total iterations: ", self.iteration)
        print("Best Solution: ", self.best.getFitness())
        print("----------------------------------*******************----------------------------------")

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print("Best so far =============")
            print("Iteration: ", self.iteration, "best: ", self.best.getFitness())
            print("=========================")

    def saveSolution(fName, solution, cost):
        file = open(fName, 'w')

        file.write(str(cost) + "\n")
        file.close()


if len(sys.argv) < 6:
    print("Error - Incorrect input")
    print("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)

problem_file = sys.argv[1]
intialization = sys.argv[2]
selection = sys.argv[3]
crossover = sys.argv[4]
mutation = sys.argv[5]
for i in range(5):
    ga = BasicTSP(problem_file, 100, 0.15, 300, intialization, selection, crossover, mutation)
    ga.search()
