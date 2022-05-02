from agent import Agent, AgentGA
from game import SnakeGameHuman, SnakeGameAI, SnakeGameGA
from genetics import GeneticAlgorithm
from helper import Plotter

from os import makedirs as os_makedirs
from os.path import exists as os_exists
from time import time as time_time
from numpy import round as np_round

from pygame import RESIZABLE as pyg_RESIZABLE
from pygame import display as pyg_display
from pygame import QUIT as pyg_QUIT
from pygame import KEYDOWN as pyg_KEYDOWN
from pygame import K_ESCAPE as pyg_K_ESCAPE
from pygame.event import get as pyg_get
from pygame.time import Clock as pyg_Clock

# Initialize the display
pyg_display.init()


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

        # Clock
        self.clock = pyg_Clock()

        # To escape a game/session early
        self.quit = False


    def run_human(self):
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


    def run_dqn(self):
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
        plotter = Plotter()
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

            # Plot data
            plotter.plot_single_agent(self.agent_scores, self.agent_mean_scores, cur_episode, self.max_episodes)

        if not os_exists(r"./models"):
            os_makedirs(r"./models")
        if not os_exists(r"./models/single_model_({}).h5".format(self.max_episodes)):
            agent.model.save(r"./models/single_model_({}).h5".format(self.max_episodes))


    def run_grl(self):
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
                    population_size = int(lines[1])
                except:
                    population_size = 20
                    print("Couldn't load population_size (run_grl)")
                try:
                    max_generations = int(lines[2])
                except:
                    max_generations = 25
                    print("Couldn't load max_generations (run_grl)")
        except:
            print("Couldn't find ManyAgentsSettings.txt (run_grl)")

        # Set internal variables
        self.population_size = population_size
        self.max_generations = max_generations
        self.simultaneous_agents = True

        # Create class objects
        self.agents = AgentGA(self.population_size)
        self.game = SnakeGameGA(self.width, self.height, self.margin, self.population_size, fps=fps)
        self.genetics = GeneticAlgorithm()
        self.plotter = Plotter()

        # Start session timer
        self.session_time = time_time()

        # Run for set number of generations (adjusting to start at gen 1)
        self._run_genetic_algorithm()

        # Save session's graph
        self.plotter.save_session()


    def _run_genetic_algorithm(self):
        ''' Process an entire population of agents. '''
        for cur_gen in range(1, self.max_generations+1):
            print(f"running {cur_gen}")
            # Reset generation data
            self.game.generation = cur_gen

            # Generation time
            self.gen_time = time_time()

            # Run for set number of generations
            self.game.play_step(self.agents)

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


    def _run_episode(self, agent):
        ''' Run an episode of the game. '''
        run = True
        while run:
            # Check for escape
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

                # Update top score if needed
                if score > self.game.top_score:
                    self.game.top_score = score
        return agent


    def _record_data(self):
        ''' Record the session data as it currently stands. '''
        self.all_scores = 0
        self.all_mean_scores = 0
        self.gen_scores = 0
        self.gen_mean_scores = 0
        self.agent_time = 0
        self.ep_time = 0
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

