from agent import AgentGA
from random import random, uniform, shuffle


class GeneticAlgorithm():
    '''The logic for getting the new generation of agents.'''
    ''' Try NEAT algorithm? neat-python '''

    def __init__(self):
        self.fitness_threshold = 0.10
        #self.fitness_threshold = 2
        self.crossover_rate = 0.05
        self.gene_size = 4
        self.mutation_rate = 0.0001
        self.mutation_degree = 0.50

        # Pool of previous parents so we can use the fittest of all time
        self.legacy_pool = None


    def _improvement_check(self, new_generation):
        '''Only allow the parents to be the absolute fittest of all generations.'''
        # For the first time, we just set it to the first generation of parents
        if self.legacy_pool == None:
            self.legacy_pool = new_generation
            for i, agent in enumerate(self.legacy_pool):
                print(f"\nAgent {i} Fitness: {agent[1]}")
                for j, layer in enumerate(agent[0].layers):
                    print(f"Layer {j}: {layer.get_weights()[0].shape}")
            print("\n\n")
        # For every other generation, we actually check for improvements
        else:
            print("\n\nImprovement Check!")
            # Reverse the lists to increase accuracy
            new_generation.reverse()
            self.legacy_pool.reverse()

            # Check for improvements
            for i in range(len(new_generation)):
                for j in range(len(self.legacy_pool)):
                    if new_generation[i][1] > self.legacy_pool[j][1]:
                        self.legacy_pool[j] = new_generation[i]
                        print(f"New Value: {self.legacy_pool[j][1]}")
                        break # so we only add a new agent once

            # Resort the legacy pool (if needed)
            self.legacy_pool.sort(key=lambda a: a[1], reverse=True)
            [print(f"Pool Fitness: {agent[1]}") for agent in self.legacy_pool]
            '''
            for i, agent in enumerate(self.legacy_pool):
                print(f"\nAgent {i} Fitness: {agent[1]}")
                for j, layer in enumerate(agent[0].layers):
                    print(f"Layer {j}: {layer.get_weights()[0].shape}")
            '''
            print("\n\n")


    def breed_population(self, population):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Get the new generation and the number of children each pair needs to have
        new_generation, num_children = population.get_parents(self.fitness_threshold)

        # Update the legacy pool of agents to include any members of the new generation
        # that are better than the old generations
        self._improvement_check(new_generation)

        # # Get the parent models
        parents = [agent[0] for agent in self.legacy_pool]
        #shuffle(parents) # Shuffle the parents into a random order

        # Initialize children
        children = AgentGA(population.population_size)

        # Crossover and mutate to get the children
        child = 0
        for i in range(0, len(parents), 2):
            for c in range(num_children):
                children.agents[child][0] = self.crossover(children.agents[child][0], parents[i], parents[i+1])
                child += 1
        return children


    def crossover(self, child, parent_one, parent_two):
        '''Apply crossover and mutation between two parents in order to get a child.'''
        # Crossover and mutate each layer for the first child
        for i in range(len(parent_one.layers)):
            # Get weights and biases of the parents
            # p1_data acts as the base for the child
            p1_data = parent_one.layers[i].get_weights()
            p2_data = parent_two.layers[i].get_weights()

            # Handle the weights
            for x in range(p1_data[0].shape[0]):
                for y in range(self.gene_size, p1_data[0].shape[1], self.gene_size):
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        p1_data[0][x][y] = p2_data[0][x][y]

                    # Check to see if mutation should occur
                    if (random() < self.mutation_rate):
                        #p1_data[0][x][y] += p1_data[0][x][y] * uniform(-self.mutation_degree, self.mutation_degree)
                        p1_data[0][x][y] += uniform(-self.mutation_degree, self.mutation_degree)

            # Handle the biases
            for x in range(p1_data[0].shape[1]):
                # Check to see if crossover should occur
                if (random() < self.crossover_rate):
                    p1_data[1][x] = p2_data[1][x]

                # Check to see if mutation should occur
                if (random() < self.mutation_rate):
                    #p1_data[1][x] += p1_data[1][x] * uniform(-self.mutation_degree, self.mutation_degree)
                    p1_data[1][x] += uniform(-self.mutation_degree, self.mutation_degree)

            # Set weights and biases in child
            child.layers[i].build(input_shape=p1_data[0].shape[0])
            child.layers[i].set_weights(p1_data)
            parent_one.layers[i].set_weights(p1_data)

        return child
