from model import LinearNet
from agent import AgentDGA
from random import random, randint, uniform, shuffle

from torch.nn import Linear as nn_Linear
from torch.nn import Parameter as nn_Parameter


class GeneticAlgorithm():
    '''The logic for getting the new generation of agents.'''
    ''' Try NEAT algorithm? neat-python '''

    def __init__(self):
        self.fitness_threshold = 0.10
        #self.fitness_threshold = 2
        self.crossover_rate = 0.10
        self.gene_size = 4
        self.mutation_rate = 0.0001
        self.mutation_degree = 0.50
        self.morph_models = False
        self.additional_info = False

        # Pool of previous parents so we can use the fittest of all time
        self.legacy_pool = None


    def _improvement_check(self, new_generation):
        '''Only allow the parents to be the absolute fittest of all generations.'''
        # For the first time, we just set it to the first generation of parents
        if self.legacy_pool == None:
            self.legacy_pool = new_generation
            
            # Check for if the model structures are changeable
            if self.morph_models: print("The models are morphing!")
            else: print("The models are static.")

            for i, agent in enumerate(self.legacy_pool):
                print(f"\nAgent {i} Fitness: {agent[1]}")
                for i, info in enumerate(agent[0].named_modules()):
                    if (i == 2) and isinstance(info[1], nn_Linear):
                        print(f"Input Hidden Layer: {list(info[1].weight.data.size())}")
                    elif (i == agent[0].num_layers+1) and isinstance(info[1], nn_Linear):
                        print(f"Output Layer: {list(info[1].weight.data.size())}")
                    elif isinstance(info[1], nn_Linear):
                        print(f"Hidden Layer: {list(info[1].weight.data.size())}")
            print("\n\n")
        # For every other generation, we actually check for improvements
        else:
            print("\n\nImprovement Check!")
            # Reverse the lists for replacing the lowest values
            new_generation.reverse()
            self.legacy_pool.reverse()

            # Check for if the model structures are changeable
            if self.morph_models: print("The models are morphing!")
            else: print("The models are static.")

            # Check for improvements
            for i in range(len(new_generation)):
                for j in range(len(self.legacy_pool)):
                    if new_generation[i][1] > self.legacy_pool[j][1]:
                        self.legacy_pool[j] = new_generation[i]
                        print(f"New Value: {self.legacy_pool[j][1]}")
                        break # so we only add a new agent once

            # Resort the legacy pool (if needed)
            self.legacy_pool.sort(key=lambda a: a[1], reverse=True)
            if self.morph_models:
                for i, agent in enumerate(self.legacy_pool):
                    print(f"\nAgent {i} Fitness: {agent[1]}")
                    for i, info in enumerate(agent[0].named_modules()):
                        if (i == 2) and isinstance(info[1], nn_Linear):
                            print(f"Input Hidden Layer: {list(info[1].weight.data.size())}")
                        elif (i == agent[0].num_layers+1) and isinstance(info[1], nn_Linear):
                            print(f"Output Layer: {list(info[1].weight.data.size())}")
                        elif isinstance(info[1], nn_Linear):
                            print(f"Hidden Layer: {list(info[1].weight.data.size())}")
            else:
                [print(f"Pool Fitness: {agent[1]}") for agent in self.legacy_pool]
            print("\n\n")


    def breed_population(self, population, shuffle_pool=True):
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
                    if self.morph_models:
                        children.agents[child][0] = self.slow_crossover(parents[i], parents[i+1], child)
                        children.agents[child+1][0] = self.slow_crossover(parents[i+1], parents[i], child+1)
                    else:
                        children.agents[child][0] = self.fast_crossover(children.agents[child][0], parents[i], parents[i+1])
                        children.agents[child+1][0] = self.fast_crossover(children.agents[child+1][0], parents[i+1], parents[i])
                    child += 2
        else:
            for i in range(0, num_parents, 2):
                for c in range(num_children):
                    children.agents[child][0] = self.fast_crossover(children.agents[child][0], parents[i], parents[i+1]) if self.morph_models else self.slow_crossover(parents[i], parents[i+1], child)
                    child += 1
        return children, num_parents


    def fast_crossover(self, child, parent1, parent2):
        '''A crossover/mutation function designed to work with static models that have the same structure.'''
        # 
        # Get all genes from parent2
        p2_genes = [] # [weights, biases]
        for layer in parent2.layers:
            # Get weight/bias data and empty lists to store genes
            weight = layer.weight.data
            bias = layer.bias.data
            weight_genes, bias_genes = [], []

            for y in range(weight.shape[1]):
                for x in range(self.gene_size, weight.shape[0], self.gene_size):
                    # Get the weight genes
                    if (weight[x][(y-self.gene_size):y].shape[0] != self.gene_size) and weight[x][(y-self.gene_size):y].shape[0]:
                        weight_genes.append(weight[x][(y-self.gene_size):y])
                    # Get the bias genes
                    if (bias[(x-self.gene_size):x].shape != self.gene_size) and bias[(x-self.gene_size):x].shape[0]:
                        bias_genes.append(bias[(x-self.gene_size):x])
            p2_genes.append([weight_genes, bias_genes])
        
        # 
        # Crossover genes
        child_crossover = []
        for i in range(parent1.num_layers):
            # Get weights and biases of the parents
            # p1_data acts as the base for the child
            weight = parent1.layers[i].weight.data
            bias = parent1.layers[i].bias.data

            #
            # Handle the weights
            for x in range(weight.shape[0]):
                for y in range(self.gene_size, weight.shape[1], self.gene_size):
                    # Check to see if crossover should occur
                    if len(p2_genes[i][0]) and (random() < self.crossover_rate):
                        weight[x][(y-self.gene_size):y] = p2_genes[i][0][int((y / weight.shape[1]) * len(p2_genes[i][0]))]

            #
            # Handle the biases
            for x in range(self.gene_size, bias.shape[0], self.gene_size):
                # Check to see if crossover should occur
                if len(p2_genes[i][1]) and (random() < self.crossover_rate):
                    bias[(x-self.gene_size):x] = p2_genes[i][1][int((x / bias.shape[0]) * len(p2_genes[i][1]))]
            
            # Collect the layer data after crossover
            child_crossover.append([weight, bias])
        
        # 
        # Mutate genes
        for i in range(len(child.layers)):
            # Copy old weight and bias values over to new model and mutate them, if it's not a new layer
            weight = child_crossover[i][0]
            bias = child_crossover[i][1]

            for x in range(weight.shape[0]):
                # Check for weight mutation
                for y in range(weight.shape[1]):
                    if (random() < self.mutation_rate):
                        weight[x][y] += uniform(-self.mutation_degree, self.mutation_degree)
                # Check for bias mutation
                if ((len(child.layers) - i) - 1) and (random() < self.mutation_rate):
                    bias[x] += uniform(-self.mutation_degree, self.mutation_degree)
                    
            # Set weights and biases in child
            child.layers[i].weight = nn_Parameter(weight)
            child.layers[i].bias = nn_Parameter(bias)
        return child


    def slow_crossover(self, parent1, parent2, child_num):
        '''A crossover/mutation function designed to work with models that can change sizes.'''
        if self.additional_info: print(f"================================\nChild {child_num}:")
        # 
        # Crossover
        #

        # Get all genes from parent2
        p2_genes = [] # [weights, biases]
        for layer in parent2.layers:
            # Get weight/bias data and empty lists to store genes
            weight = layer.weight.data
            bias = layer.bias.data
            weight_genes, bias_genes = [], []

            for y in range(weight.shape[1]):
                for x in range(self.gene_size, weight.shape[0], self.gene_size):
                    # Get the weight genes
                    if (weight[x][(y-self.gene_size):y].shape[0] != self.gene_size) and weight[x][(y-self.gene_size):y].shape[0]:
                        weight_genes.append(weight[x][(y-self.gene_size):y])
                    # Get the bias genes
                    if (bias[(x-self.gene_size):x].shape != self.gene_size) and bias[(x-self.gene_size):x].shape[0]:
                        bias_genes.append(bias[(x-self.gene_size):x])
            p2_genes.append([weight_genes, bias_genes])

        # Crossover genes
        child_crossover = []
        for i in range(parent1.num_layers):
            # Get weights and biases of the parents
            # p1_data acts as the base for the child
            weight = parent1.layers[i].weight.data
            bias = parent1.layers[i].bias.data

            # The layer we use for p2, since they might have different numbers of layers
            p2_layer = int(i * parent2.num_layers / parent1.num_layers)

            #
            # Handle the weights
            for x in range(weight.shape[0]):
                for y in range(self.gene_size, weight.shape[1], self.gene_size):
                    # Check to see if crossover should occur
                    if len(p2_genes[p2_layer][0]) and (random() < self.crossover_rate):
                        weight[x][(y-self.gene_size):y] = p2_genes[p2_layer][0][int((y / weight.shape[1]) * len(p2_genes[p2_layer][0]))]

            #
            # Handle the biases
            for x in range(self.gene_size, bias.shape[0], self.gene_size):
                # Check to see if crossover should occur
                if len(p2_genes[p2_layer][1]) and (random() < self.crossover_rate):
                    bias[(x-self.gene_size):x] = p2_genes[p2_layer][1][int((x / bias.shape[0]) * len(p2_genes[p2_layer][1]))]
            
            # Collect the layer data after crossover
            child_crossover.append([weight, bias])

        # 
        # Mutate
        #

        # Value lists
        modded_layer = [False for i in range(len(child_crossover))]
        hidden_layers = []

        #
        # Mutate number of neurons
        for i in range(len(child_crossover) - 1):
            num_neurons = child_crossover[i][0].shape[0]
            # Check to see if the size of this layer will mutate
            if (random() < self.mutation_rate):
                num_neurons += 1 if (random() > 0.5) else -1
                if self.additional_info: print("Neuron count changed!")
            hidden_layers.append(num_neurons)

        #
        # Mutate number of hidden layers
        if (random() < self.mutation_rate):
            # Remove layer
            if len(hidden_layers) and (random() > 0.5):
                # Choose layer to remove
                location = randint(0, len(hidden_layers)-1)
                del hidden_layers[location]
                # We've removed it, so we don't want to try to copy it
                modded_layer.insert(location, True)
                if self.additional_info: print("Removed hidden layer!")
            # Add layer
            else:
                # Choose where to insert the new layer and how many neurons it should have
                location = randint(0, len(hidden_layers))
                num_neurons = randint(1, 10)
                # Insert layer
                hidden_layers.insert(location, num_neurons)
                modded_layer.insert(location, True)
                if self.additional_info: print("Added hidden layer!")

        #
        # Copy weights and biases, then mutate individual weights and biases
        child = LinearNet(parent1.input_size, parent1.output_size, hidden_layers=hidden_layers, random_model=False)
        p_counter = 0
        for i in range(len(child.layers)):
            # Copy old weight and bias values over to new model and mutate them, if it's not a new layer
            weight = child.layers[i].weight.data
            bias = child.layers[i].bias.data
            if not modded_layer[i]:
                _x = weight.shape[0] if weight.shape[0] < child_crossover[p_counter][0].shape[0] else child_crossover[p_counter][0].shape[0]
                _y = weight.shape[1] if weight.shape[1] < child_crossover[p_counter][0].shape[1] else child_crossover[p_counter][0].shape[1]
                weight[0:_x, 0:_y] = child_crossover[p_counter][0][0:_x, 0:_y]
                bias[0:_x] = child_crossover[p_counter][1][0:_x]

                for x in range(_x):
                    # Check for weight mutation
                    for y in range(_y):
                        if (random() < self.mutation_rate):
                            weight[x][y] += uniform(-self.mutation_degree, self.mutation_degree)
                        
                    # Check for bias mutation
                    if ((len(child.layers) - i) - 1) and (random() < self.mutation_rate):
                        bias[x] += uniform(-self.mutation_degree, self.mutation_degree)
                p_counter += 1
            # Set weights and biases in child
            child.layers[i].weight = nn_Parameter(weight)
            child.layers[i].bias = nn_Parameter(bias)
        
        if self.additional_info:
            for i, info in enumerate(child.named_modules()):
                if (i == 2) and isinstance(info[1], nn_Linear):
                    print(f"\nInput Hidden Layer: {list(info[1].weight.data.size())}")
                elif (i == child.num_layers+1) and isinstance(info[1], nn_Linear):
                    print(f"Output Layer: {list(info[1].weight.data.size())}\n")
                    print("================================")
                elif isinstance(info[1], nn_Linear):
                    print(f"Hidden Layer: {list(info[1].weight.data.size())}")

        return child
