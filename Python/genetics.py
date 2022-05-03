from agent import AgentGA
from random import random, uniform, shuffle


class GeneticAlgorithm():
    def __init__(self):
        #self.fitness_threshold = 0.10
        self.fitness_threshold = 2
        self.crossover_rate = 0.05
        self.mutation_rate = 0.10
        self.mutation_degree = 0.50

        # Pool of previous parents so we can use the fittest of all time
        self.legacy_pool = None
    

    def _improvement_check(self, new_generation):
        ''' Only allow the parents to be the absolute fittest of all generations. '''
        # For the first time, we just set it to the first generation of parents
        changed = False
        if self.legacy_pool == None:
            self.legacy_pool = new_generation
            changed = True
        # For every other generation, we actually check for improvements
        else:
            # Reverse the lists to increase accuracy
            new_generation.reverse()
            self.legacy_pool.reverse()

            # Check for improvements
            for i in range(len(new_generation)):
                for j in range(len(self.legacy_pool)):
                    if new_generation[i][1] > self.legacy_pool[j][1]:
                        self.legacy_pool[j] = new_generation[i]
                        changed = True
                        print(f"New Value: {self.legacy_pool[j][1]}")
                        break # so we only add a new agent once
            
            # Resort the legacy pool (if needed)
            if changed: self.legacy_pool.sort(key=lambda a: a[1], reverse=True)
            #[print(f"Pool Fitness: {agent[1]}") for agent in self.legacy_pool]
            for i, agent in enumerate(self.legacy_pool):
                print(f"\nAgent {i} Fitness: {agent[1]}")
                for j, layer in enumerate(agent[0].layers):
                    print(f"Layer {j}: {layer.get_weights()[0].shape}")
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
        count = 0
        for c in range(1, num_children-1, 2):
            for i in range(1, len(parents)-1, 2):
                children.agents[count][0], children.agents[count+1][0] = self.crossover(parents[i], parents[i+1])
                count += 2
        return children


    def crossover(self, parent_one, parent_two):
        ''' Apply crossover and mutation between two parents in order to get a child. '''
        # Find which of the two models has more layers
        if len(parent_one.layers) > len(parent_two.layers):
            max_layers = len(parent_one.layers)
            min_layers = len(parent_two.layers)
        else:
            max_layers = len(parent_two.layers)
            min_layers = len(parent_one.layers)

        # Crossover and mutate each layer for the first child
        for i in range(max_layers):
            if i > min_layers: break

            # Get weights and biases of the parents
            # p1 acts as the base for the child
            child1_data = parent_one.layers[i].get_weights()
            p2_data = parent_two.layers[i].get_weights()

            # Find which of the two layers is bigger
            X = child1_data[0].shape[0] if child1_data[0].shape[0] > p2_data[0].shape[0] else p2_data[0].shape[0]
            Y = child1_data[0].shape[1] if child1_data[0].shape[1] > p2_data[0].shape[1] else p2_data[0].shape[1]
            
            # Handle the weights
            for x in range(X):
                for y in range(Y):
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        child1_data[0][x][y] = p2_data[0][x][y]
                        
                    # Check to see if mutation should occur
                    if (random() < self.mutation_rate):
                        child1_data[0][x][y] += child1_data[0][x][y] * uniform(-self.mutation_degree, self.mutation_degree)

            # Handle the biases
            for x in range(X):
                # Check to see if crossover should occur
                if (random() < self.crossover_rate):
                    child1_data[1][x] = p2_data[1][x]
                
                # Check to see if mutation should occur
                if (random() < self.mutation_rate):
                    child1_data[1][x] += child1_data[1][x] * uniform(-self.mutation_degree, self.mutation_degree)

            # Set weights and biases in child
            #child.layers[i].build(input_shape=child1_data[0].shape[0])
            #child.layers[i].set_weights(child1_data)
            parent_one[i].set_weights(child1_data)

        # Crossover and mutate each layer for the second child
        for i in range(min_layers):
            if i > max_layers: break

            # Get weights and biases of the parents
            # p1 acts as the base for the child
            p1_data = parent_one.layers[i].get_weights()
            child2_data = parent_two.layers[i].get_weights()

            # Find which of the two layers is bigger
            X = child2_data[0].shape[0] if child2_data[0].shape[0] > p1_data[0].shape[0] else p1_data[0].shape[0]
            Y = child2_data[0].shape[1] if child2_data[0].shape[1] > p1_data[0].shape[1] else p1_data[0].shape[1]
            
            # Handle the weights
            for x in range(X):
                for y in range(Y):
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        child2_data[0][x][y] = p1_data[0][x][y]
                        
                    # Check to see if mutation should occur
                    if (random() < self.mutation_rate):
                        child2_data[0][x][y] += child2_data[0][x][y] * uniform(-self.mutation_degree, self.mutation_degree)

            # Handle the biases
            for x in range(X):
                # Check to see if crossover should occur
                if (random() < self.crossover_rate):
                    child2_data[1][x] = p1_data[1][x]
                
                # Check to see if mutation should occur
                if (random() < self.mutation_rate):
                    child2_data[1][x] += child2_data[1][x] * uniform(-self.mutation_degree, self.mutation_degree)

            # Set weights and biases in child
            #child.layers[i].build(input_shape=child2_data[0].shape[0])
            #child.layers[i].set_weights(child2_data)
            parent_two[i].set_weights(child2_data)

        #return child
        return parent_one, parent_two
