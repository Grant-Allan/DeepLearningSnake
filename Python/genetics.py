from agent import AgentGA
from random import random


class GeneticAlgorithm():
    def breed_population(self, population, fitness_threshold=0.10, crossover_rate=0.50, mutation_rate=0.02, mutation_degree=0.1, mutate=True):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Get the parent models and
        parents, num_children = population.get_parents(fitness_threshold=fitness_threshold)

        # Initialize children
        children = AgentGA(population.population_size)

        # Breed population
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

            # Check to see if crossover should occur in the weights and biases
            for x, w_or_b in enumerate(child_weights):
                for y in range(len(w_or_b)):
                    if (random() < crossover_rate):
                        # If p1 is the base...
                        if parent1:
                            # Crossover
                            child_weights[x][y] = p2_weights[x][y]
                        # If p2 is the base...
                        else:
                            # Crossover
                            child_weights[x][y] = p1_weights[x][y]
                
                    # Check to see if/where mutations should occur
                    if mutate:
                        if (random() < mutation_rate):
                            if (random() > 0.50):
                                p1_weights[x][y] += p1_weights[x][y] * mutation_degree
                            else:
                                p1_weights[x][y] -= p1_weights[x][y] * mutation_degree
            
            # Set weights and biases in child
            child.layers[i].build(input_shape=child_weights[0].shape[0])
            child.layers[i].set_weights(child_weights)
        return child
