from agent import Agent
import random
import operator


class GeneticAlgorithm():
    def breed_population(self, agents, fittness_threshold=0.10, crossover_rate=0.50, mutation_rate=0.05, mutation_degree=0.05, mutate=True):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Sort by highest to lowest score
        agents.sort(key=operator.attrgetter("top_score"), reverse=True)

        # Get the number of breeding agents
        pop_size = len(agents)
        cutoff = (int)(fittness_threshold * pop_size)
        if not cutoff % 2: cutoff -= 1
        if cutoff < 2: return agents

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
        for i in range(len(parent_one.layers)):
            # Get weights and biases
            p1_weights = parent_one.layers[i].get_weights()
            p2_weights = parent_two.layers[i].get_weights()
            
            # Cycle through layer's weights
            for x, row in enumerate(p1_weights[0]):
                for y, _ in enumerate(row):
                    # Apply crossover
                    if (random.random() < crossover_rate):
                        p1_weights[0][x][y] = p2_weights[0][x][y]
                    
                    # Apply mutation
                    if mutate:
                        if (random.random() < mutation_rate):
                            if (random.random() > 0.50):
                                p1_weights[0][x][y] += p1_weights[0][x][y] * mutation_degree
                            else:
                                p1_weights[0][x][y] -= p1_weights[0][x][y] * mutation_degree

            # Cycle through layer's biases
            for b in range(len(p1_weights[1])):
                    # Apply crossover
                    if (random.random() < crossover_rate):
                        p1_weights[1][b] = p2_weights[1][b]
                    
                    # Apply mutation
                    if mutate:
                        if (random.random() < mutation_rate):
                            if (random.random() > 0.50):
                                p1_weights[1][b] += p1_weights[1][b] * mutation_degree
                            else:
                                p1_weights[1][b] -= p1_weights[1][b] * mutation_degree
            
            # Set weights and biases in child
            child.layers[i].build(input_shape=p1_weights[0].shape[0])
            child.layers[i].set_weights(p1_weights)
        return child
