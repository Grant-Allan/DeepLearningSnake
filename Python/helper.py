from enum import Enum
from collections import namedtuple

from os import makedirs as os_makedirs
from os import system as os_system
from os import name as os_name
from os.path import exists as os_exists

from matplotlib.pyplot import ion as plt_ion
from matplotlib.pyplot import subplots as plt_subplots
from matplotlib.pyplot import subplots_adjust as plt_subplots_adjust
from matplotlib.pyplot import show as plt_show
from matplotlib.pyplot import pause as plt_pause
from matplotlib.pyplot import savefig as plt_savefig


# Set up a clear console function
clearConsole = lambda: os_system('cls' if os_name in ('nt', 'dos') else 'clear')

# Directions dictionary so we can use .RIGHT, .LEFT, etc instead of numbers
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Let's us use .x and .y instead of [0] or [1]
Point = namedtuple("Point", ["x", "y"])

# Game values
TILE_SIZE = 20

# Agent values
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# RGB colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
SLATE_GRAY = (112, 128, 144)
DIM_GRAY = (105, 105, 105)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN1 = (0, 100, 0)
GREEN2 = (0, 150, 0)
GREEN3 = (0, 200, 0)


class Plotter():
    ''' Class to hold all plotting functions. '''
    def __init__(self, single_agent=False):
        plt_ion() # turn on interactable plots
        if not single_agent:
            fig, self.ax = plt_subplots(2, 2) # initialize 2x2 graphs
            self.ax[1, 1].axis("off")
            fig.set_size_inches(8, 6) # set figure size

            # set the spacing between subplots
            plt_subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
        else:
            _, self.ax = plt_subplots()


    def plot_data(self, all_scores, all_mean_scores, gen_scores, gen_mean_scores,
                  gen_num, agent_scores, agent_mean_scores, agent_num, num_agents,
                  cur_episode, num_episodes, top_score):
        ''' Plot the data for the game. '''
        # Set internal values
        self.all_scores = all_scores
        self.all_mean_scores = all_mean_scores

        self.gen_scores = gen_scores
        self.gen_mean_scores = gen_mean_scores
        self.gen_num = gen_num

        self.agent_scores = agent_scores
        self.agent_mean_scores = agent_mean_scores
        self.agent_num = agent_num

        self.num_agents = num_agents
        self.cur_episode = cur_episode
        self.num_episodes = num_episodes
        self.top_score = top_score

        # Plot data
        self._plot_aggregate()
        self._plot_generation()
        self._plot_agent()
        self._show_data()

        # Show data
        plt_show(block=False)
        plt_pause(0.001)


    def _plot_aggregate(self):
        ''' Plot all data collected by the game. '''
        # Clear previous display
        self.ax[0, 0].cla()

        # Set title and axes labels
        self.ax[0, 0].set_title("Total Aggregate Data")
        self.ax[0, 0].set_xlabel("Number of Games")
        self.ax[0, 0].set_ylabel("Score")

        # Plot data and set legend
        self.ax[0, 0].plot(self.all_scores, label="Scores")
        self.ax[0, 0].plot(self.all_mean_scores, label="Mean Scores")
        self.ax[0, 0].legend(loc="upper left", prop={'size': 8})

        # Set number at tip of each line declaring the current value
        self.ax[0, 0].text(len(self.all_scores)-1, self.all_scores[-1], str(self.all_scores[-1]))
        self.ax[0, 0].text(len(self.all_mean_scores)-1, self.all_mean_scores[-1], str(self.all_mean_scores[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax[0, 0].set_xlim(xmin=0)
        self.ax[0, 0].set_ylim(ymin=0)


    def _plot_generation(self):
        ''' Plot the data for the current generation. '''
        # Clear previous display
        self.ax[0, 1].cla()

        # Set title and axes labels
        self.ax[0, 1].set_title(f"Generation {self.gen_num} Data")
        self.ax[0, 1].set_xlabel("Number of Games")
        self.ax[0, 1].set_ylabel("Score")

        # Plot data and set legend
        self.ax[0, 1].plot(self.gen_scores, label="Scores")
        self.ax[0, 1].plot(self.gen_mean_scores, label="Mean Scores")
        self.ax[0, 1].legend(loc="upper left", prop={'size': 8})

        # Set number at tip of each line declaring the current value
        self.ax[0, 1].text(len(self.gen_scores)-1, self.gen_scores[-1], str(self.gen_scores[-1]))
        self.ax[0, 1].text(len(self.gen_mean_scores)-1, self.gen_mean_scores[-1], str(self.gen_mean_scores[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax[0, 1].set_xlim(xmin=0)
        self.ax[0, 1].set_ylim(ymin=0)


    def _plot_agent(self):
        ''' Plot the data for the current agent. '''
        # Clear previous display
        self.ax[1, 0].cla()

        # Set title and axes labels
        self.ax[1, 0].set_title(f"Agent {self.agent_num} Data")
        self.ax[1, 0].set_xlabel("Number of Games")
        self.ax[1, 0].set_ylabel("Score")

        # Plot data and set legend
        self.ax[1, 0].plot(self.agent_scores, label="Scores")
        self.ax[1, 0].plot(self.agent_mean_scores, label="Mean Scores")
        self.ax[1, 0].legend(loc="upper left", prop={'size': 8})

        # Set number at tip of each line declaring the current value
        self.ax[1, 0].text(len(self.agent_scores)-1, self.agent_scores[-1], str(self.agent_scores[-1]))
        self.ax[1, 0].text(len(self.agent_mean_scores)-1, self.agent_mean_scores[-1], str(self.agent_mean_scores[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax[1, 0].set_xlim(xmin=0)
        self.ax[1, 0].set_ylim(ymin=0)


    def _show_data(self):
        ''' Show the data. '''
        # Clear previous display
        self.ax[1, 1].cla()

        # Turn off axes display
        self.ax[1, 1].axis("off")

        # Set title
        self.ax[1, 1].set_title("Game Data")

        # Display text
        self.ax[1, 1].text(0, 0.4,
                f"Current Agent {self.agent_num} of {self.num_agents}\n" +
                f"Current Episode: {self.cur_episode} of {self.num_episodes}\n" +
                f"Current Generation: {self.gen_num}\n" +
                f"Current Score: {self.agent_scores[-1]}\n" +
                f"Top Score: {self.top_score}",
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 10},
                size=14)


    def plot_single_agent(self, scores, mean_scores, cur_ep, num_eps):
        ''' Plot the data when running a sessions with just one agent. '''
        # Clear previous display
        self.ax.cla()

        # Set title and axes labels
        self.ax.set_title(f"Score Tracker\nEpisode {cur_ep} of {num_eps}")
        self.ax.set_xlabel("Number of Games")
        self.ax.set_ylabel("Score")

        # Plot data and set legend
        self.ax.plot(scores, label="Scores")
        self.ax.plot(mean_scores, label="Mean Scores")
        self.ax.legend(loc="upper left")

        # Set number at tip of each line declaring the current value
        self.ax.text(len(scores)-1, scores[-1], str(scores[-1]))
        self.ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax.set_xlim(xmin=0)
        self.ax.set_ylim(ymin=0)

        # Show data
        plt_show(block=False)
        plt_pause(0.001)


    def save_session(self):
        ''' Save the data for the entire session. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        plt_savefig(r"./graphs/session_graph.jpg")


    def save_gen(self, gen_num):
        ''' Save the data for this generation. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        if not os_exists(r"./graphs/generation_graphs"):
            os_makedirs(r"r./graphs/generation_graphs")
        plt_savefig(r"./graphs/generation_graphs/graph_gen{}.jpg".format(gen_num))


    def save_agent(self, gen_num, agent_num):
        ''' Save the data for this agent. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        if not os_exists(r"./graphs/agent_graphs/gen{}".format(gen_num)):
            os_makedirs(r"./graphs/agent_graphs/gen{}".format(gen_num))
        plt_savefig(r"./graphs/agent_graphs/graph_agent{}.jpg".format(agent_num))
