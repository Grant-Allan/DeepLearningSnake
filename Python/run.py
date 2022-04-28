from agent import Agent
from game import SnakeGameHuman, SnakeGameAI
from genetics import GeneticAlgorithm
from helper import Plotter

from os import makedirs as os_makedirs
from os.path import exists as os_exists
from time import time as time_time
from numpy import round as np_round

from pygame import QUIT as pyg_QUIT
from pygame import KEYDOWN as pyg_KEYDOWN
from pygame import K_ESCAPE as pyg_K_ESCAPE
from pygame.event import get as pyg_get



class RunGame():
    '''
    Controller class for running the snake game as a human,
    for training a single DQN snake, or a population of DQN
    snakes in tandem with a deep genetic algorithm.
    '''
    def __init__(self, width, height, margin):
        # Initialize input data
        self.width = width
        self.height = height
        self.margin = margin

        # To escape a game/session early
        self.quit = False


    def run_human(self, fps=10):
        ''' Run the snake game in a way that a human can play. '''
        # Get data
        try:
            with open(r"./Resources/StandardGameSettings.txt") as file:
                lines = file.readlines()
                try:
                    fps = int(lines[0])
                except:
                    fps = 10
                    print("Couldn't load fps (run_human)")
        except:
            print("Couldn't find StandardGameSettings.txt (run_dqn)")

        # Create game object
        game = SnakeGameHuman(self.width, self.height, self.margin, fps=fps)

        # Game loop
        while True:
            # Run game step
            game_over, score = game.play_step()

            # Check for death
            if game_over: break
        print(f"\nFinal Score: {score}\n")


    def run_dqn(self, fps=100, max_episodes=120):
        ''' Run a single deep Q learning snake. '''
        # Get data
        try:
            with open(r"./Resources/SingleAgentSettings.txt") as file:
                lines = file.readlines()
                try:
                    fps = int(lines[0])
                except:
                    fps = 100
                    print("Couldn't load fps (run_dqn)")
                try:
                    max_episodes = int(lines[1])
                except:
                    max_episodes = 120
                    print("Couldn't load max_episodes (run_dqn)")
        except:
            print("Couldn't find SingleAgentSettings.txt (run_dqn)")

        # Set internal variables
        self.max_episodes = max_episodes

        # Create objects
        agent = Agent()
        plotter = Plotter(single_agent=True)
        self.game = SnakeGameAI(self.width, self.height, self.margin, fps=fps)

        # Set colors
        self.game.color1 = agent.color1
        self.game.color2 = agent.color2

        # Set aggregate data lists
        self.agent_scores, self.agent_mean_scores = [], []

        # Set score aggregate for this agent
        self.agent_score = 0

        # Run for set number of episodes (adjusting to start at ep 1)
        for cur_episode in range(1, self.max_episodes+1):
            # Episode variables
            agent.episode = cur_episode
            self.game.agent_episode = cur_episode

            # Episode loop
            agent = self._run_episode(agent, single_agent=True)

            # Check for escape
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    self.quit = True
                elif event.type == pyg_KEYDOWN:
                    if event.key == pyg_K_ESCAPE:
                        self.quit = True
            if self.quit: break

            # Plot data
            plotter.plot_single_agent(self.agent_scores, self.agent_mean_scores, cur_episode, self.max_episodes)

        if not os_exists(r"./models"):
            os_makedirs(r"./models")
        if not os_exists(r"./models/single_model_({}).h5".format(self.max_episodes)):
            agent.model.save(r"./models/single_model_({}).h5".format(self.max_episodes))


    def run_grl(self, fps=100, population_size=20, max_episodes=10, max_generations=25):
        ''' Run a session of genetic reinforcement learning. '''
        # Get data
        try:
            with open(r"./Resources/ManyAgentsSettings.txt") as file:
                lines = file.readlines()
                try:
                    fps = int(lines[0])
                except:
                    fps = 100
                    print("Couldn't load fps (run_grl)")
                try:
                    max_episodes = int(lines[1])
                except:
                    max_episodes = 10
                    print("Couldn't load max_episodes (run_grl)")
                try:
                    population_size = int(lines[2])
                except:
                    population_size = 20
                    print("Couldn't load population_size (run_grl)")
                try:
                    max_generations = int(lines[3])
                except:
                    max_generations = 25
                    print("Couldn't load max_generations (run_grl)")
        except:
            print("Couldn't find ManyAgentsSettings.txt (run_grl)")

        # Set internal variables
        self.population_size = population_size
        self.max_episodes = max_episodes
        self.max_generations = max_generations

        # Create class objects
        self.agents = [Agent() for i in range(self.population_size)]
        self.game = SnakeGameAI(self.width, self.height, self.margin, fps=fps)
        self.genetics = GeneticAlgorithm()
        self.plotter = Plotter()

        # Set aggregate data lists
        self.all_scores, self.all_mean_scores = [], []
        self.all_episodes = 0

        # Start session timer
        self.session_time = time_time()

        # Run for set number of generations (adjusting to start at gen 1)
        self._run_generation()

        # Save session's graph
        self.plotter.save_session()


    def _run_generation(self):
        ''' Process an entire population of agents. '''
        for cur_gen in range(1, self.max_generations+1):
            # Reset generation data
            self.game.generation = cur_gen

            # Set score aggregate for this generation
            self.gen_score = 0

            # Reset generation data lists
            self.gen_scores, self.gen_mean_scores = [], []
            self.gen_episodes = 0

            # Generation time
            self.gen_time = time_time()

            # Run for set number of generations
            self._run_agent()

            # Check for escape
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    self.quit = True
                elif event.type == pyg_KEYDOWN:
                    if event.key == pyg_K_ESCAPE:
                        self.quit = True
            if self.quit: break

            # Save generation's graph
            self.plotter.save_gen(cur_gen)

            # Make new population
            self.agents = self.genetics.breed_population(self.agents)


    def _run_agent(self):
        ''' Run an agent, whether on it's own or as part of a population. '''
        for agent_num, agent in enumerate(self.agents):
            # Set colors
            self.game.color1 = agent.color1
            self.game.color2 = agent.color2

            # Set agent number
            self.game.agent_num = agent_num+1

            # Reset agent data lists
            self.agent_scores, self.agent_mean_scores = [], []

            # Set score aggregate for this agent
            self.agent_score = 0

            # Agent time
            self.agent_time = time_time()

            # Run for set number of episodes (adjusting to start at ep 1)
            for cur_episode in range(1, self.max_episodes+1):
                # Episode time
                self.ep_time = time_time()

                # Episode variables
                agent.episode = cur_episode
                self.game.agent_episode = cur_episode
                self.all_episodes += 1
                self.gen_episodes += 1

                # Episode loop
                agent = self._run_episode(agent)

                # Check for escape
                for event in pyg_get():
                    # Check for exiting out of window
                    if event.type == pyg_QUIT:
                        self.quit = True
                    elif event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_ESCAPE:
                            self.quit = True
                if self.quit: break

                # Record data
                self._record_data()

            # Update the agent in the population
            self.agents[agent_num] = agent

            # Check for esapce
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    self.quit = True
                elif event.type == pyg_KEYDOWN:
                    if event.key == pyg_K_ESCAPE:
                        self.quit = True
            if self.quit: break

        # Save agent's graph
        self.plotter.save_agent(self.game.generation, agent_num)


    def _run_episode(self, agent, single_agent=False):
        ''' Run an episode of the game. '''
        run = True
        while run:
            # Check for esapce
            for event in pyg_get():
                # Check for exiting out of window
                if event.type == pyg_QUIT:
                    self.quit = True
                elif event.type == pyg_KEYDOWN:
                    if event.key == pyg_K_ESCAPE:
                        self.quit = True
            if self.quit: break

            # Get old state
            state_old = agent.get_state(self.game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = self.game.play_step(final_move)
            state_new = agent.get_state(self.game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # Snake died
            if done:
                run = False

                # Train long memory
                self.game.reset()
                agent.train_long_memory()

                # Update agent's internal score if needed
                if score > agent.top_score:
                    agent.top_score = score

                if score > self.game.top_score:
                    self.game.top_score = score

                if not single_agent:
                    # Save model if it's the best (and update top score)
                    if score > self.game.top_score:
                        if not os_exists(r"./models"):
                            os_makedirs(r"./models")
                        if not os_exists(r"./models/pop{}-eps{}-gens{}".format(self.population_size, self.max_episodes, self.max_generations)):
                            agent.model.save(r"./models/pop{}-eps{}-gens{}".format(self.population_size, self.max_episodes, self.max_generations))

                        if not os_exists(r"./models/model_gen{}_({}-{}-{}).h5".format(self.game.generation, self.population_size, self.max_episodes, self.max_generations)):
                            agent.model.save(r"./models/model_gen{}_({}-{}-{}).h5".format(self.game.generation, self.population_size, self.max_episodes, self.max_generations))

                    # Update aggregate data
                    self.game.total_score += score
                    self.game.mean_score = np_round((self.game.total_score / self.all_episodes), 3)
                    self.all_scores.append(score)
                    self.all_mean_scores.append(self.game.mean_score)

                    # Updte generation data
                    self.gen_score += score
                    gen_mean = np_round((self.gen_score / self.gen_episodes), 3)
                    self.gen_scores.append(score)
                    self.gen_mean_scores.append(gen_mean)

                # Update agent data
                self.agent_score += score
                agent_mean = np_round((self.agent_score / self.game.agent_episode), 3)
                self.agent_scores.append(score)
                self.agent_mean_scores.append(agent_mean)
        return agent


    def _record_data(self):
        ''' Record the session data as it currently stands. '''
        self.plotter.plot_data(self.all_scores,
                               self.all_mean_scores,
                               self.gen_scores,
                               self.gen_mean_scores,
                               self.game.generation,
                               self.agent_scores,
                               self.agent_mean_scores,
                               self.game.agent_num,
                               self.max_generations,
                               len(self.agents),
                               self.game.agent_episode,
                               self.max_episodes,
                               self.game.top_score,
                               time_time()-self.session_time,
                               time_time()-self.gen_time,
                               time_time()-self.agent_time,
                               time_time()-self.ep_time)

