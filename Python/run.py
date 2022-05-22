from agent import AgentDQN, AgentGA
from game import SnakeGameHuman, SnakeGameDQN, SnakeGameGA
from genetics import GeneticAlgorithm
from helper import Plotter, WIDTH, HEIGHT, MARGIN

from os import makedirs as os_makedirs
from os import remove as os_remove
from os.path import exists as os_exists
from time import time as time_time
from numpy import round as np_round

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
    def __init__(self):
        # Initialize input data
        self.width = WIDTH
        self.height = HEIGHT
        self.margin = MARGIN

        # Clock
        self.clock = pyg_Clock()

        # To escape a game/session early
        self.quit = False


    def run_human(self):
        ''' Run the snake game in a way that a human can play. '''
        # Get data
        try:
            with open(r"./Resources/HumanGameSettings.txt") as file:
                lines = file.readlines()
                try:
                    fps = int(lines[0])
                except:
                    fps = 10
                    print("Couldn't load fps (run_human)")
        except:
            print("Couldn't find HumanGameSettings.txt (run_human)")

        # Create game object
        game = SnakeGameHuman(fps=fps)

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
            with open(r"./Resources/DQN_Settings.txt") as file:
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
            print("Couldn't find DQN_Settings.txt (run_dqn)")

        # Set internal variables
        self.max_episodes = max_episodes
        num_agents = 10

        # Create objects
        agent = AgentDQN()
        plotter = Plotter()
        self.game = SnakeGameDQN(fps=fps)

        # Set colors
        self.game.color1 = agent.color1
        self.game.color2 = agent.color2

        # Start session timer
        self.session_time = time_time()

        for a in range(num_agents):
            # Set aggregate data lists
            self.agent_scores, self.agent_mean_scores = [], []

            # Set score aggregate for this agent
            self.agent_score = 0

            # Start agent timer
            self.agent_time = time_time()

            # Run for set number of episodes (adjusting to start at ep 1)
            for cur_episode in range(1, self.max_episodes+1):
                # Start episode timer
                self.episode_time = time_time()

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
                plotter.plot_DQN(a,
                                 self.agent_scores,
                                 self.game.top_score,
                                 self.agent_mean_scores,
                                 cur_episode,
                                 self.max_episodes,
                                 time_time()-self.session_time,
                                 time_time()-self.agent_time,
                                 time_time()-self.episode_time,
                                 agent.model.layers)


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

                # Update agent and game's internal score if needed
                if score > agent.top_score:
                    if not os_exists(r"./models"):
                        os_makedirs(r"./models")

                    shapes = [f"({layer.get_weights()[0].shape})" for layer in agent.model.layers]
                    shapes = '-'.join(shapes)

                    if not os_exists(r"./models/DQN_model_({})_({})__({}).h5".format(self.max_episodes, shapes, agent.top_score)):
                        agent.model.save(r"./models/DQN_model_({})_({})__({}).h5".format(self.max_episodes, shapes, agent.top_score))
                    else: # delete existing file to make a new one
                        os_remove(r"./models/DQN_model_({})_({})__({}).h5".format(self.max_episodes, shapes, agent.top_score))
                        agent.model.save(r"./models/DQN_model_({})_({})__({}).h5".format(self.max_episodes, shapes, agent.top_score))
                    agent.top_score = score

                # Train long memory
                self.game.reset()
                agent.train_long_memory()

                # Update aggregate score lists
                self.agent_scores.append(score)
                agent.total_score += score
                agent.mean_score = np_round((agent.total_score / len(self.agent_scores)), 3)
                self.agent_mean_scores.append(agent.mean_score)
        return agent


    def run_dga(self):
        ''' Run a deep genetic algorithm session. '''
        # Get data
        try:
            with open(r"./Resources/DGA_Settings.txt") as file:
                lines = file.readlines()
                try:
                    fps = int(lines[0])
                except:
                    fps = 50
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
            print("Couldn't find DGA_Settings.txt (run_grl)")

        # Set internal variables
        self.population_size = population_size
        self.max_generations = max_generations

        # Create class objects
        self.agents = AgentGA(self.population_size)
        self.game = SnakeGameGA(self.population_size, self.max_generations, fps=fps)
        self.genetics = GeneticAlgorithm()
        #self.plotter = Plotter()

        # Start session timer
        self.session_time = time_time()

        # Run for set number of generations (adjusting to start at gen 1)
        self._run_genetic_algorithm()

        # Save session's graph
        #self.plotter.save_session()


    def _run_genetic_algorithm(self):
        ''' Process an entire population of agents. '''
        for cur_gen in range(1, self.max_generations+1):
            # Reset generation data
            self.game.generation = cur_gen

            # Generation time
            self.gen_time = time_time()

            # Generation run loop
            game_over = False
            while not game_over:
                # Run for set number of generations
                game_over, self.agents = self.game.play_step(self.agents)

                # Check for escape
                for event in pyg_get():
                    # Check for exiting out of window
                    if event.type == pyg_QUIT:
                        self.quit = True
                    elif event.type == pyg_KEYDOWN:
                        if event.key == pyg_K_ESCAPE:
                            self.quit = True
                if self.quit: break
            if self.quit: break

            # Reset the internal data in preparation for the next generation
            self.game.reset()

            # Make new population
            self.agents = self.genetics.breed_population(self.agents)
