    
    
class Run():
    '''
    Controller class for running the snake game as a human,
    for training a single DQN snake, or a population of DQN
    snakes in tandem with a deep genetic algorithm.
    '''

    def run_human(self):
        pass


    def run_dqn(self):
        pass

    def run_grl(self, population_size=20, max_episodes=10, max_generations=10):
        ''' Run a session of genetic reinforcement learning. '''
        agents = [Agent() for i in range(population_size)]
        game = SnakeGameAI()
        genetics = GeneticAlgorithm()

        scores, mean_scores = [], []
        all_scores, all_mean_scores, gen_scores, gen_mean_scores, agent_scores, agent_mean_scores = [], [], [], [], [], []
        
        for cur_gen in range(1, max_generations+1):
            self.run_generations()
            
    

    def run_generation(self):
        # Reset generation data
        game.generation = cur_gen
        gen_scores, gen_mean_scores = [], []

        # Make new population
        agents = genetics.breed_population(agents)

        # Save generation's graph
        save_graph(cur_gen)

        for agent_num, agent in enumerate(agents):
            self.run_agent()

            

    def run_agent(self):
        
        # Set colors
        game.color1 = agent.color1
        game.color2 = agent.color2

        # Set agent number
        game.agent_num = agent_num

        # Reset agent data lists
        agent_scores, agent_mean_scores = [], []

        total_score = 0

        for cur_episode in range(1, max_episodes+1):
            self.run_episodes()
                

    
    def run_episode(self):
        agent.episode = cur_episode
        game.agent_episode = cur_episode
        run = True
        while run:
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # Snake died
            if done:
                run = False
                # Train long memory, plot result
                game.reset()
                agent.episode = cur_episode
                game.agent_episode = cur_episode
                agent.train_long_memory()

                # Save model if it's the best (and update top score)
                if score > game.top_score:
                    if not os.path.exists("./models"):
                        os.makedirs("./models")
                    agent.model.save(f"./models/model_gen{cur_gen}.h5")
                    game.top_score = score

                # Update agent's internal score if needed
                if score > agent.top_score:
                    agent.top_score = score

                total_score += score
                game.mean_score = np.round((total_score / cur_episode), 3)

                

    def record_data():
        # Record data
        scores.append(score)
        mean_scores.append(game.mean_score)
        
        plot_data(all_scores, all_mean_scores, gen_scores, gen_mean_scores, cur_gen, agent_scores, agent_mean_scores, game.agent_num)
        print(f"Agent {game.agent_num}")
        print(f"Populatino {len(agents)}")
        print(f"Episode: {cur_episode}")
        print(f"Generation: {cur_gen}")
        print(f"Score: {score}")
        print(f"Top Score: {game.top_score}")
        print(f"Mean: {game.mean_score}\n")