from tkinter.tix import Y_REGION
from turtle import hideturtle
from model import LinearNet
from agent import AgentDGA
from random import random, randint, uniform, shuffle


class GeneticAlgorithm():
    '''The logic for getting the new generation of agents.'''
    ''' Try NEAT algorithm? neat-python '''

    def __init__(self):
        self.fitness_threshold = 0.10
        #self.fitness_threshold = 2
        self.crossover_rate = 0.05
        self.gene_size = 4
        self.mutation_rate = 0.001
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
            for i, agent in enumerate(self.legacy_pool):
                print(f"\nAgent {i} Fitness: {agent[1]}")
                for j, layer in enumerate(agent[0].layers):
                    print(f"Layer {j}: {layer.get_weights()[0].shape}")
            print("\n\n")


    def breed_population(self, population, shuffle_pool=True, morph_models=True):
        '''
        Crossover the weights and biases of the fittest members of the population,
        then randomly mutate weights and biases.
        '''
        # Get the new generation and the number of children each pair needs to have
        new_generation, num_children, num_parents = population.get_parents(self.fitness_threshold)

        # Update the legacy pool of agents to include any members of the new generation
        # that are better than the old generations
        self._improvement_check(new_generation)

        # # Get the parent models
        parents = [agent[0] for agent in self.legacy_pool]
        if shuffle_pool: shuffle(parents) # Shuffle the parents into a random order

        # Initialize children
        children = AgentDGA(population.population_size)

        # Crossover and mutate to get the children
        child = 0
        if shuffle_pool:
            for i in range(0, num_parents, 2):
                for c in range(0, num_children, 2):
                    if morph_models:
                        children.agents[child][0] = self.slow_crossover(parents[i], parents[i+1])
                        children.agents[child+1][0] = self.slow_crossover(parents[i+1], parents[i])
                    else:
                        children.agents[child][0] = self.fast_crossover(children.agents[child][0], parents[i], parents[i+1])
                        children.agents[child+1][0] = self.fast_crossover(children.agents[child+1][0], parents[i+1], parents[i])
                    child += 2
        else:
            for i in range(0, num_parents, 2):
                for c in range(num_children):
                    children.agents[child][0] = self.fast_crossover(children.agents[child][0], parents[i], parents[i+1]) if morph_models else self.slow_crossover(parents[i], parents[i+1])
                    child += 1
        return children, num_parents


    def fast_crossover(self, child, parent1, parent2):
        '''Apply crossover and mutation between two parents in order to get a child.'''
        # Crossover and mutate each layer
        for i in range(len(parent1.layers)):
            # Get weights and biases of the parents
            # p1_data acts as the base for the child
            p1_data = parent1.layers[i].get_weights()
            p2_data = parent2.layers[i].get_weights()

            # Handle the weights
            for x in range(p1_data[0].shape[0]):
                for y in range(self.gene_size, p1_data[0].shape[1], self.gene_size):
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        p1_data[0][x][(y-self.gene_size):y] = p2_data[0][x][(y-self.gene_size):y]

                    # Check to see if mutation should occur
                    if (random() < self.mutation_rate):
                        #p1_data[0][x][y] += p1_data[0][x][y] * uniform(-self.mutation_degree, self.mutation_degree)
                        p1_data[0][x][y] += uniform(-self.mutation_degree, self.mutation_degree)

                    # Handle the biases
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        p1_data[1][(y-self.gene_size):y] = p2_data[1][(y-self.gene_size):y]

                    # Check to see if mutation should occur
                    if (random() < self.mutation_rate):
                        #p1_data[1][y] += p1_data[1][y] * uniform(-self.mutation_degree, self.mutation_degree)
                        p1_data[1][y] += uniform(-self.mutation_degree, self.mutation_degree)

            # Set weights and biases in child
            child.layers[i].build(input_shape=p1_data[0].shape[0])
            child.layers[i].set_weights(p1_data)
        return child


    def slow_crossover(self, parent1, parent2):
        '''A crossover/mutation function designed to work with models that can change sizes.'''
        # 
        # Crossover

        # Ratio of number of model layers between parents
        layer_ratio = len(parent1.layers) / len(parent2.layers)

        # Get all genes from parent2
        # This will prevent the model from trying to take a gene section from parent2 but it being the wrong size
        p2_genes = [] # weights, biases
        p2_shapes = []
        for layer in parent2.layers:
            p2_data = layer.get_weights()
            p2_shapes.append(p2_data[0].shape)
            weights = []
            biases = []

            # Get the weight genes
            for x in range(p2_data[0].shape[0]):
                for y in range(self.gene_size, p2_data[0].shape[1], self.gene_size):
                        weights.append(p2_data[0][x][(y-self.gene_size):y])
            
            # Get the bias genes
            for x in range(self.gene_size, p2_data[1].shape[0], self.gene_size):
                biases.append(p2_data[1][(x-self.gene_size):x])

            p2_genes.append([weights, biases])

        # Crossover genes
        child_crossover = []
        for i in range(len(parent1.layers)):
            # Get weights and biases of the parents
            # p1_data acts as the base for the child
            p1_data = parent1.layers[i].get_weights()

            p2_layer = int(i * layer_ratio)

            # Handle the weights
            for x in range(p1_data[0].shape[0]):
                for y in range(self.gene_size, p1_data[0].shape[1], self.gene_size):
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        p1_data[0][x][(y-self.gene_size):y] = p2_genes[p2_layer][0][int((y / p1_data[0].shape[1]) * len(p2_genes[p2_layer][0]))]

                    # Handle the biases
                    # Check to see if crossover should occur
                    if (random() < self.crossover_rate):
                        p1_data[1][(y-self.gene_size):y] = p2_genes[p2_layer][1][int((y / p1_data[1].shape[0]) * len(p2_genes[p2_layer][1]))]
            
            # Collect the layer data after crossover
            child_crossover.append(p1_data)

        # 
        # Mutate
        modded_layer = [False for i in range(len(child_crossover))]
        hidden_layers = []
        pieces = [] # like puzzle pieces; the sizes that make it work


        # Mutate number of neurons
        for i in range(len(child_crossover) - 1):
            num_neurons = child_crossover[i][0].shape[1]
            
            if (random() < self.mutation_rate):
                num_neurons += 1 if (random() > 0.5) else -1
            
            # Get the x value of the layer
            if len(pieces) and (child_crossover[i-1][0].shape[1] >= pieces[-1][1]):
                x = pieces[-1][1]
            elif len(pieces) and (child_crossover[i-1][0].shape[1] < pieces[-1][1]):
                x = child_crossover[i-1][0].shape[1]
            # On the first hidden layer, it's just the input shape
            else:
                x = child_crossover[i][0].shape[0]

            # Get the y value of the layer
            y = num_neurons if (child_crossover[i][0].shape[1] > num_neurons) else child_crossover[i][0].shape[1]

            pieces.append([x, y])
            hidden_layers.append(num_neurons)
        
         # Add the output layer where the output size can't be mutated
        if len(pieces) and (child_crossover[-2][0].shape[1] >= pieces[-1][1]):
            pieces.append([pieces[-1][1], 3])
        elif len(pieces) and (child_crossover[-2][0].shape[1] < pieces[-1][1]):
            pieces.append([child_crossover[-2][0].shape[1], 3])
        else:
            pieces.append([child_crossover[-1][0].shape[0], 3])


        # Mutate number of hidden layers
        if (random() < self.mutation_rate):
            # Remove layer
            if len(hidden_layers) and (random() > 0.5):
                # Choose layer to remove
                location = randint(0, len(hidden_layers)-1)
                del hidden_layers[location]

                # We've removed it, so we don't want to try to copy it
                modded_layer.insert(location, True)

                # Adjust input/output after preceeding and following layers in pieces
                if len(hidden_layers):
                    pieces[location+1][0] = pieces[location][1]
                # Just input/output layer; basic perceptron
                else:
                    pieces[location][0] = child_crossover[0][0].shape[0]

            # Add layer
            else:
                # Choose where to insert the new layer and how many neurons it should have
                location = randint(0, len(hidden_layers))
                num_neurons = randint(1, 10)

                # Insert layer
                hidden_layers.insert(location, num_neurons)
                modded_layer.insert(location, True)

                # Modify pieces as needed
                if pieces[location][0] > num_neurons: pieces[location][0] = num_neurons
                if location: pieces[location-1][1] = num_neurons


        # Copy weights and biases, then mutate individual weights and biases
        child = LinearNet.linear_QNet(child_crossover[0][0].shape[0], child_crossover[-1][0].shape[1], hidden_layers=hidden_layers, random_model=False)
        p_counter = 0
        for i in range(len(child.layers)):
            # Copy old weight and bias values over to new model and mutate them, if it's not a new layer
            child_data = child.layers[i].get_weights()
            if not modded_layer[i]:
                child_data[0][0:pieces[p_counter][0], 0:pieces[p_counter][1]] = child_crossover[p_counter][0][0:pieces[p_counter][0], 0:pieces[p_counter][1]]
                child_data[1][0:pieces[p_counter][1]] = child_crossover[p_counter][1][0:pieces[p_counter][1]]

                for x in range(pieces[p_counter][0]):
                    # Check for weight mutation
                    for y in range(pieces[p_counter][1]):
                        if (random() < self.mutation_rate):
                            child_data[0][x][y] += uniform(-self.mutation_degree, self.mutation_degree)
                        
                        # Check for bias mutation
                        if (random() < self.mutation_rate):
                            child_data[1][y] += uniform(-self.mutation_degree, self.mutation_degree)
                
                p_counter += 1
            # Set weights and biases in child
            child.layers[i].build(input_shape=child_data[0].shape[0])
            child.layers[i].set_weights(child_data)

        return child
