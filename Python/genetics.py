from agent import Agent
from random import random
import operator


class GeneticAlgorithm():
    def breed_population(self, agents, two_fittest=False, fittness_threshold=0.10, crossover_rate=0.50, mutation_rate=0.02, mutation_degree=0.1, mutate=True):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Sort by highest to lowest score
        agents.sort(key=operator.attrgetter("top_score"), reverse=True)

        # Get the number of breeding agents
        pop_size = len(agents)
        if not two_fittest:
            cutoff = (int)(fittness_threshold * pop_size)
            if not cutoff % 2: cutoff -= 1
            if cutoff < 2: return agents
        else:
            cutoff = 2

        # Get number of times each parent pair needs to breed
        num_children = pop_size // cutoff

        # Initialize children
        children = [Agent() for i in range(pop_size)]

        # Breed population
        for i in range(0, cutoff, 2):
            for c in range(num_children):
                children[i*c].model = self.crossover(children[i*c].model, agents[i].model, agents[i+1].model, crossover_rate, mutation_rate, mutation_degree, mutate)

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

            # Check to see if crossover should occur in the weights and bias
            for j in range(len(child_weights)):
                for x in range(len(child_weights[0])):
                    if (random() < crossover_rate):
                        # If p1 is the base...
                        if parent1:
                            # Crossover
                            child_weights[j][x] = p2_weights[j][x]
                        # If p2 is the base...
                        else:
                            # Crossover
                            child_weights[j][x] = p1_weights[j][x]
                
                    # Check to see if/where mutations should occur
                    if mutate:
                        for y in range(len(child_weights[j][x])):
                            if (random() < mutation_rate):
                                if (random() > 0.50):
                                    p1_weights[j][x][y] += p1_weights[j][x][y] * mutation_degree
                                else:
                                    p1_weights[j][x][y] -= p1_weights[j][x][y] * mutation_degree
            
            # Set weights and biases in child
            child.layers[i].build(input_shape=child_weights[0].shape[0])
            child.layers[i].set_weights(child_weights)
        return child
