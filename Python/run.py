from agent import Agent
from genetics import GeneticAlgorithm
from game import StartMenu, SnakeGameAI, SnakeGameHuman
from helper import Plotter

from os import makedirs as os_makedirs
from os.path import exists as os_exists
from numpy import round as np_round


class RunGame():
    '''
    Controller class for running the snake game as a human,
    for training a single DQN snake, or a population of DQN
    snakes in tandem with a deep genetic algorithm.
    '''
    def main_menu(self):
        '''
        A main menu to open the game into, where you can select
        your game mode and change various settings.
        '''
        start_menu = StartMenu()


    def run_human(self, fps=10):
        ''' Run the snake game in a way that a human can play. '''
        # Create game object
        game = SnakeGameHuman(fps=fps)

        # Game loop
        while True:
            game_over, score = game.play_step()
            if game_over: break
        print(f"\nFinal Score: {score}\n")


    def run_dqn(self, fps=100, max_episodes=100):
        ''' Run a single deep Q learning snake. '''
        # Set internal variables
        self.max_episodes = max_episodes

        # Create objects
        agent = Agent()
        plotter = Plotter(single_agent=True)
        self.game = SnakeGameAI(fps=fps)

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
            run = True
            while run:
                run = self._run_episode(agent, single_agent=True)

            # Plot data
            plotter.plot_single_agent(self.agent_scores, self.agent_mean_scores)



    def run_grl(self, fps=100, population_size=20, max_episodes=10, max_generations=10):
        ''' Run a session of genetic reinforcement learning. '''
        # Set internal variables
        self.population_size = population_size
        self.max_episodes = max_episodes
        self.max_generations = max_generations

        # Create class objects
        self.agents = [Agent() for i in range(self.population_size)]
        self.game = SnakeGameAI(fps=fps)
        self.genetics = GeneticAlgorithm()
        self.plotter = Plotter()

        # Set aggregate data lists
        self.all_scores, self.all_mean_scores = [], []
        self.all_episodes = 0

        # Run for set number of generations (adjusting to start at gen 1)
        for cur_gen in range(1, self.max_generations+1):
            self._run_generation(cur_gen)

        # Save session's graph
        self.plotter.save_session()


    def _run_generation(self, cur_gen):
        ''' Process an entire population of agents. '''
        # Reset generation data
        self.game.generation = cur_gen

        # Set score aggregate for this generation
        self.gen_score = 0

        # Reset generation data lists
        self.gen_scores, self.gen_mean_scores = [], []
        self.gen_episodes = 0

        # Run for set number of generations
        for agent_num, agent in enumerate(self.agents):
            self._run_agent(agent_num, agent)

        # Save generation's graph
        self.plotter.save_gen(cur_gen)

        # Make new population
        self.agents = self.genetics.breed_population(self.agents)


    def _run_agent(self, agent_num, agent):
        ''' Run an agent, whether on it's own or as part of a population. '''
        # Set colors
        self.game.color1 = agent.color1
        self.game.color2 = agent.color2

        # Set agent number
        self.game.agent_num = agent_num+1

        # Reset agent data lists
        self.agent_scores, self.agent_mean_scores = [], []

        # Set score aggregate for this agent
        self.agent_score = 0

        # Run for set number of episodes (adjusting to start at ep 1)
        for cur_episode in range(1, self.max_episodes+1):
            # Episode variables
            agent.episode = cur_episode
            self.game.agent_episode = cur_episode
            self.all_episodes += 1
            self.gen_episodes += 1

            # Episode loop
            run = True
            while run:
                run = self._run_episode(agent)

            # Record data
            self._record_data()

        # Save agent's graph
        self.plotter.save_agent(self.game.generation, agent_num)



    def _run_episode(self, agent, single_agent=False):
        ''' Run an episode of the game. '''
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

            if not single_agent:
                # Save model if it's the best (and update top score)
                if score > self.game.top_score:
                    if not os_exists("./models"):
                        os_makedirs("./models")
                    if not os_exists(f"./models/model_gen{self.game.generation}_({self.population_size}-{self.max_episodes}-{self.max_generations}).h5"):
                        agent.model.save(f"./models/model_gen{self.game.generation}_({self.population_size}-{self.max_episodes}-{self.max_generations}).h5")
                    self.game.top_score = score

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

        # Snake lives
        else:
            run = True
        return run


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
                               len(self.agents),
                               self.game.agent_episode,
                               self.max_episodes,
                               self.game.top_score)
