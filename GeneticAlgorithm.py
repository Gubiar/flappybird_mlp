import numpy as np

from Demo import FlappyBird_Human
from MLP import MLP

class GeneticAlgorithm:
    def __init__(self, entrada, oculta, saida, taxaDeAprendizado = 0.1):
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.taxaDeAprendizado = taxaDeAprendizado
        self.populationSize = 100
        self.mutationRate = 0.05
        self.numberOfGenerations = 100
        self.stopEarlyCriteria = 10
        self.population = [MLP(self.entrada, self.oculta, self.saida, self.taxaDeAprendizado) for _ in range(self.populationSize)]
        self.bestIndividual = None
        self.bestFitness = -np.inf

    def fitnessFunction(self, individual, game):
        score = game.runWithAi(individual)
        return score

    def evaluateIndividual(self, individual, game):
        fitness = self.fitnessFunction(individual, game)
        return fitness

    def selectParents(self, game):
        fitnesses = [self.evaluateIndividual(individual, game) for individual in self.population]
        parents = np.random.choice(self.population, size=2, p=fitnesses/np.sum(fitnesses))
        return parents

    def generateChildren(self, parents):
        child1 = MLP(self.entrada, self.oculta, self.saida, self.taxaDeAprendizado)
        child2 = MLP(self.entrada, self.oculta, self.saida, self.taxaDeAprendizado)
        
        # Crossover de um ponto nos pesos da entrada para a camada oculta
        crossover_point = np.random.randint(0, self.entrada*self.oculta)
        child1.pesos_entrada_oculta = np.concatenate((parents[0].pesos_entrada_oculta[:crossover_point], parents[1].pesos_entrada_oculta[crossover_point:]))
        child2.pesos_entrada_oculta = np.concatenate((parents[1].pesos_entrada_oculta[:crossover_point], parents[0].pesos_entrada_oculta[crossover_point:]))

        # Crossover de um ponto nos pesos da camada oculta para a saída
        crossover_point = np.random.randint(0, self.oculta*self.saida)
        child1.pesos_oculta_saida = np.concatenate((parents[0].pesos_oculta_saida[:crossover_point], parents[1].pesos_oculta_saida[crossover_point:]))
        child2.pesos_oculta_saida = np.concatenate((parents[1].pesos_oculta_saida[:crossover_point], parents[0].pesos_oculta_saida[crossover_point:]))

        return [child1, child2]

    def mutationOperator(self, individual):
        # Mutação nos pesos da entrada para a camada oculta
        mutation_mask = np.random.rand(*individual.pesos_entrada_oculta.shape) < self.mutationRate
        individual.pesos_entrada_oculta += mutation_mask * np.random.normal(size=individual.pesos_entrada_oculta.shape)

        # Mutação nos pesos da camada oculta para a saída
        mutation_mask = np.random.rand(*individual.pesos_oculta_saida.shape) < self.mutationRate
        individual.pesos_oculta_saida += mutation_mask * np.random.normal(size=individual.pesos_oculta_saida.shape)

    def execute(self):
        game = FlappyBird_Human()
        
        for generation in range(self.numberOfGenerations):
            new_population = []
            for _ in range(self.populationSize // 2):
                parents = self.selectParents(game)
                children = self.generateChildren(parents)
                new_population.extend(children)

            for individual in new_population:
                self.mutationOperator(individual)

            fitnesses = [self.evaluateIndividual(individual, game) for individual in new_population]
            best_fitness_index = np.argmax(fitnesses)
            
            if fitnesses[best_fitness_index] > self.bestFitness:
                self.bestFitness = fitnesses[best_fitness_index]
                self.bestIndividual = new_population[best_fitness_index]

            print(f"Generation {generation}, Best Fitness {self.bestFitness}")

            if generation - best_fitness_index > self.stopEarlyCriteria:
                break

            self.population = new_population