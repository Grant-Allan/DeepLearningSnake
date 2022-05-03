from agent import AgentGA
from random import random


class GeneticAlgorithm():
    def breed_population(self, population, fitness_threshold=0.10, crossover_rate=0.50, mutation_rate=0.02, mutation_degree=0.1, mutate=True):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Get the parent models and number of children each pair needs to have
        parents, num_children = population.get_parents(fitness_threshold=fitness_threshold)

        # Initialize children
        children = AgentGA(population.population_size)

        # Breed population
        '''
        i = 0
        c = 0
        children.agents[i*c][0] = self.crossover(children.agents[i*c][0], parents[i], parents[i+1], crossover_rate, mutation_rate, mutation_degree, mutate)
        '''
        for i in range(0, len(parents), 2):
            for c in range(num_children):
                children.agents[i*c][0] = self.crossover(children.agents[i*c][0], parents[i], parents[i+1], crossover_rate, mutation_rate, mutation_degree, mutate)

        return children


    def crossover(self, child, parent_one, parent_two, crossover_rate, mutation_rate, mutation_degree, mutate):
        ''' Apply crossover and mutation between two parents in order to get a child. '''
        # Crossover and mutate each layer
        for i in range(len(child.layers)):
            # Get weights and biases of the parents
            p1_weights = parent_one.layers[i].get_weights()
            p2_weights = parent_two.layers[i].get_weights()

            # Decide who serves as the base for the child
            if (random() < 0.50):
                child_weights = p1_weights
                parent1 = True
            else:
                child_weights = p2_weights
                parent1 = False
            
            # Handle the weights
            for x in range(child_weights[0].shape[0]):
                for y in range(child_weights[0].shape[1]):
                    # Check for crossover
                    if (random() < crossover_rate):
                        # If p1 is the base...
                        if parent1:
                            child_weights[0][x][y] = p2_weights[0][x][y]
                        # If p2 is the base...
                        else:
                            child_weights[0][x][y] = p1_weights[0][x][y]
                        
                    # Check to see if/where mutations should occur
                    if mutate:
                        if (random() < mutation_rate):
                            if (random() > 0.50):
                                child_weights[0][x][y] += child_weights[0][x][y] * mutation_degree
                            else:
                                child_weights[0][x][y] -= child_weights[0][x][y] * mutation_degree

            # Handle the biases
            for x in range(child_weights[1].shape[0]):
                # Check for crossover
                if (random() < crossover_rate):
                    # If p1 is the base...
                    if parent1:
                        child_weights[1][x] = p2_weights[1][x]
                    # If p2 is the base...
                    else:
                        child_weights[1][x] = p1_weights[1][x]
                
                # Check to see if/where mutations should occur
                if mutate:
                    if (random() < mutation_rate):
                        if (random() > 0.50):
                            child_weights[1][x] += child_weights[1][x] * mutation_degree
                        else:
                            child_weights[1][x] -= child_weights[1][x] * mutation_degree

            # Set weights and biases in child
            child.layers[i].build(input_shape=child_weights[0].shape[0])
            child.layers[i].set_weights(child_weights)
        return child
