from enum import Enum
from collections import namedtuple

from os import makedirs as os_makedirs
from os.path import exists as os_exists

from IPython.display import clear_output as dis_clear_output
from IPython.display import display as dis_display

from matplotlib.pyplot import ion as plt_ion
from matplotlib.pyplot import subplots as plt_subplots
from matplotlib.pyplot import gcf as plt_gcf
from matplotlib.pyplot import clf as plt_clf
from matplotlib.pyplot import show as plt_show
from matplotlib.pyplot import pause as plt_pause
from matplotlib.pyplot import savefig as plt_savefig
from matplotlib.pyplot import title as plt_title
from matplotlib.pyplot import xlabel as plt_xlabel
from matplotlib.pyplot import ylabel as plt_ylabel
from matplotlib.pyplot import plot as plt_plot
from matplotlib.pyplot import legend as plt_legend
from matplotlib.pyplot import text as plt_text
from matplotlib.pyplot import xlim as plt_xlim
from matplotlib.pyplot import ylim as plt_ylim


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
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


class Plotter():
    ''' Class to hold all plotting functions. '''
    def __init__(self, single_agent=False):
        plt_ion() # turn on interactable plots
        if not single_agent: _, self.ax = plt_subplots(2, 2) # initialize 2x2 graphs

    def plot_data(self, all_scores, all_mean_scores, gen_scores, gen_mean_scores, gen_num, agent_scores, agent_mean_scores, agent_num):
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

        # Clear previous display
        dis_clear_output(wait=True)
        dis_display(plt_gcf())
        #clearConsole()
        plt_clf()

        # Plot data
        self._plot_aggregate()
        self._plot_generation()
        self._plot_agent()

        # Show data
        plt_show(block=False)
        plt_pause(0.001)


    def _plot_aggregate(self):
        ''' Plot all data collected by the game. '''
        self.ax[0, 0].title("Total Aggregate Data")
        self.ax[0, 0].xlabel("Number of Games")
        self.ax[0, 0].ylabel("Score")

        self.ax[0, 0].plot(self.all_scores, label="Scores")
        self.ax[0, 0].plot(self.all_mean_scores, label="Mean Scores")
        self.ax[0, 0].legend(loc="upper left")

        self.ax[0, 0].text(len(self.all_scores)-1, self.all_scores[-1], str(self.all_scores[-1]))
        self.ax[0, 0].text(len(self.all_mean_scores)-1, self.all_mean_scores[-1], str(self.all_mean_scores[-1]))

        self.ax[0, 0].xlim(xmin=0)
        self.ax[0, 0].ylim(ymin=0)


    def _plot_generation(self):
        ''' Plot the data for the current generation. '''
        self.ax[0, 1].title(f"Generation {self.gen_num} Data")
        self.ax[0, 1].xlabel("Number of Games")
        self.ax[0, 1].ylabel("Score")

        self.ax[0, 1].plot(self.gen_scores, label="Scores")
        self.ax[0, 1].plot(self.gen_mean_scores, label="Mean Scores")
        self.ax[0, 1].legend(loc="upper left")

        self.ax[0, 1].text(len(self.gen_scores)-1, self.gen_scores[-1], str(self.gen_scores[-1]))
        self.ax[0, 1].text(len(self.gen_mean_scores)-1, self.gen_mean_scores[-1], str(self.gen_mean_scores[-1]))

        self.ax[0, 1].xlim(xmin=0)
        self.ax[0, 1].ylim(ymin=0)


    def _plot_agent(self):
        ''' Plot the data for the current agent. '''
        self.ax[1, 0].title(f"Agent {self.agent_num} Data")
        self.ax[1, 0].xlabel("Number of Games")
        self.ax[1, 0].ylabel("Score")

        self.ax[1, 0].plot(self.agent_scores, label="Scores")
        self.ax[1, 0].plot(self.agent_mean_scores, label="Mean Scores")
        self.ax[1, 0].legend(loc="upper left")

        self.ax[1, 0].text(len(self.agent_scores)-1, self.agent_scores[-1], str(self.agent_scores[-1]))
        self.ax[1, 0].text(len(self.agent_mean_scores)-1, self.agent_mean_scores[-1], str(self.agent_mean_scores[-1]))

        self.ax[1, 0].xlim(xmin=0)
        self.ax[1, 0].ylim(ymin=0)


    def plot_single_agent(self, scores, mean_scores):
        ''' Plot the data when running a sessions with just one agent. '''
        # Clear previous display
        dis_clear_output(wait=True)
        dis_display(plt_gcf())
        #clearConsole()
        plt_clf()

        # Plot data
        plt_title(f"Score Tracker")
        plt_xlabel("Number of Games")
        plt_ylabel("Score")

        plt_plot(scores, label="Scores")
        plt_plot(mean_scores, label="Mean Scores")
        plt_legend(loc="upper left")

        plt_text(len(scores)-1, scores[-1], str(scores[-1]))
        plt_text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

        plt_xlim(xmin=0)
        plt_ylim(ymin=0)

        # Show data
        plt_show(block=False)
        plt_pause(0.001)

    
    def save_session(self):
        ''' Save the data for the entire session. '''
        if not os_exists("./graphs"):
            os_makedirs("./graphs")
        plt_savefig("./graphs/session_graph.jpg")


    def save_gen(self, gen_num):
        ''' Save the data for this generation. '''
        if not os_exists("./graphs"):
            os_makedirs("./graphs")
        if not os_exists("./graphs/generation_graphs"):
            os_makedirs("./graphs/generation_graphs")
        plt_savefig(f"./graphs/generation_graphs/graph_gen{gen_num}.jpg")
    

    def save_gen(self, gen_num, agent_num):
        ''' Save the data for this agent. '''
        if not os_exists("./graphs"):
            os_makedirs("./graphs")
        if not os_exists(f"./graphs/agent_graphs/gen{gen_num}"):
            os_makedirs(f"./graphs/agent_graphs/gen{gen_num}")
        plt_savefig(f"./graphs/agent_graphs/graph_agent{agent_num}.jpg")
