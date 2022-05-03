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
WIDTH = TILE_SIZE*38
HEIGHT = TILE_SIZE*28
MARGIN = TILE_SIZE*4

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
    def __init__(self):
        plt_ion() # turn on interactable plots

        fig, self.ax = plt_subplots(2) # initialize 2x2 graphs
        self.ax[1].axis("off") # turn axes off for the right graph
        #fig.set_size_inches(8, 6) # set figure size

        # set the spacing between subplots
        plt_subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.7)
    

    def plot_DQN(self, scores, top_score, mean_scores, cur_ep, num_eps, session_time_elapsed, episode_time_elapsed):
        ''' Plot the data when running a sessions with just one agent. '''
        
        # Update data
        self._plot_data_DQN(scores, mean_scores, cur_ep, num_eps)
        self._show_data_DQN(scores[-1], top_score, mean_scores[-1], cur_ep, num_eps, session_time_elapsed, episode_time_elapsed)

        # Show data
        plt_show(block=False)
        plt_pause(0.001)

    
    def _plot_data_DQN(self, scores, mean_scores, cur_ep, num_eps):
        # Clear previous display
        self.ax[0].cla()

        # Set title and axes labels
        self.ax[0].set_title(f"Score Tracker\nEpisode {cur_ep} of {num_eps}")
        self.ax[0].set_xlabel("Number of Games")
        self.ax[0].set_ylabel("Score")

        # Plot data and set legend
        self.ax[0].plot(scores, label="Scores")
        self.ax[0].plot(mean_scores, label="Mean Scores")
        self.ax[0].legend(loc="upper left")

        # Set number at tip of each line declaring the current value
        self.ax[0].text(len(scores)-1, scores[-1], str(scores[-1]))
        self.ax[0].text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax[0].set_xlim(xmin=0)
        self.ax[0].set_ylim(ymin=0)


    def _show_data_DQN(self, cur_score, top_score, cur_mean, cur_ep, num_eps, session_time_elapsed, episode_time_elapsed):
        ''' Show the data. '''
        # Clear previous display
        self.ax[1].cla()

        # Turn off axes display
        self.ax[1].axis("off")

        # Set title
        self.ax[1].set_title("Session Data")

        # Display text
        self.ax[1].text(0.50, 0.40,
                f"Current Episode: {cur_ep} of {num_eps}\n" +
                f"Current Score: {cur_score}\n" +
                f"Current Mean: {cur_mean}\n" +
                f"Current Top Score: {top_score}\n\n" +

                f"(hours:minutes:seconds)\n" +
                f"Episode Time: {int(episode_time_elapsed//3600)}:{int(episode_time_elapsed//60 % 60)}:{int(episode_time_elapsed % 60)}\n" +
                f"Session Time: {int(session_time_elapsed//3600)}:{int(session_time_elapsed//60 % 60)}:{int(session_time_elapsed % 60)}",

                bbox={"facecolor": "white", "alpha": 1, "pad": 10},
                ha="center",
                va="center",
                size=12)


    def save_session(self):
        ''' Save the data for the entire session. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        if not os_exists(r"./graphs/session_graph.jpg"):
            plt_savefig(r"./graphs/session_graph.jpg")
