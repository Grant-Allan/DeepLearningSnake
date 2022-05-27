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
TILE_SIZE = 40
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

        self.fig, self.ax = plt_subplots(1, 2) # initialize 2x2 graphs
        self.ax[1].axis("off") # turn axes off for the right graph
        self.fig.set_size_inches(10, 5) # set figure size

        # set the spacing between subplots
        plt_subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.9)


    def plot_DQN(self,
                 agent,
                 scores,
                 top_score,
                 mean_scores,
                 cur_ep,
                 num_eps,
                 session_time_elapsed,
                 agent_time_elapsed,
                 episode_time_elapsed,
                 layers):
        ''' Plot the data when running a sessions with just one agent. '''

        # Update data
        self._plot_data_DQN(agent, scores, mean_scores, cur_ep, num_eps)
        self._show_data_DQN(scores[-1], top_score, mean_scores[-1], cur_ep, num_eps,
                            session_time_elapsed, agent_time_elapsed, episode_time_elapsed,
                            layers)

        # Show data
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.0000000000001)


    def _plot_data_DQN(self, agent, scores, mean_scores, cur_ep, num_eps):
        # Clear previous display
        self.ax[0].cla()

        # Set title and axes labels
        self.ax[0].set_title(f"Agent {agent}\nEpisode {cur_ep} of {num_eps}")
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


    def _show_data_DQN(self, cur_score, top_score, cur_mean, cur_ep, num_eps,
                       session_time_elapsed, agent_time_elapsed, episode_time_elapsed,
                       layers):
        ''' Show the data. '''
        # Clear previous display
        self.ax[1].cla()

        # Turn off axes display
        self.ax[1].axis("off")

        # Set title
        self.ax[1].set_title("Session Data")

        # Get model data
        shapes = [f"Hidden Layer: {layer.get_weights()[0].shape}" if i != len(layers)-1 else f"Output Layer: {layer.get_weights()[0].shape}" for i, layer in enumerate(layers)]
        shapes[0] = f"Input Layer: {layers[0].get_weights()[0].shape}"

        # Display text
        self.ax[1].text(0.50, 0.50,
                f"Current Episode: {cur_ep} of {num_eps}\n\n" +

                f"Top Score: {top_score}\n" +
                f"Agent Score: {cur_score}\n" +
                f"Agent Mean: {cur_mean}\n\n" +

                f"Episode Time: {int(episode_time_elapsed//3600)}h {int(episode_time_elapsed//60 % 60)}m {int(episode_time_elapsed % 60)}s\n" +
                f"Agent Time: {int(agent_time_elapsed//3600)}h {int(agent_time_elapsed//60 % 60)}m {int(agent_time_elapsed % 60)}s\n" +
                f"Session Time: {int(session_time_elapsed//3600)}h {int(session_time_elapsed//60 % 60)}m {int(session_time_elapsed % 60)}s\n\n" +

                f"Model:\n" +
                '\n'.join(shapes),

                bbox={"facecolor": "white", "alpha": 1, "pad": 10},
                ha="center",
                va="center",
                size=12)


    def save_dqn_session(self):
        ''' Save the data for the entire session. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        if not os_exists(r"./graphs/DQN_session_graph.jpg"):
            plt_savefig(r"./graphs/DQN_session_graph.jpg")
    

    def plot_DGA(self,
                 cur_gen,
                 num_gens,
                 scores,
                 all_mean_scores,
                 gen_mean_score,
                 pop_size,
                 num_parents,
                 top_gen_score,
                 top_score,
                 total_mean,
                 gen_mean,
                 session_time_elapsed,
                 gen_time_elapsed):
        '''Plot and display all data for the DGA session.'''
        # Update data
        self._plot_data_DGA(cur_gen, num_gens, scores, all_mean_scores, gen_mean_score)
        self._show_data_DGA(cur_gen, num_gens, pop_size, num_parents,
                            top_gen_score, top_score,
                            total_mean, gen_mean,
                            session_time_elapsed, gen_time_elapsed)

        # Show data
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.0000000000001)


    def _plot_data_DGA(self, cur_gen, num_gens, scores, all_mean_scores, gen_mean_score):
        '''Plot the dga data.'''
        # Clear previous display
        self.ax[0].cla()

        # Set title and axes labels
        self.ax[0].set_title(f"Generation {cur_gen} of {num_gens}")
        self.ax[0].set_xlabel("Number of Generations")
        self.ax[0].set_ylabel("Score")

        # Plot data and set legend
        self.ax[0].plot(scores, label="Scores")
        self.ax[0].plot(all_mean_scores, label="All Mean Scores")
        self.ax[0].plot(gen_mean_score, label="Generation Mean Scores")
        self.ax[0].legend(loc="upper left")

        # Set number at tip of each line declaring the current value
        self.ax[0].text(len(scores)-1, scores[-1], str(scores[-1]))
        self.ax[0].text(len(all_mean_scores)-1, all_mean_scores[-1], str(all_mean_scores[-1]))
        self.ax[0].text(len(gen_mean_score)-1, gen_mean_score[-1], str(gen_mean_score[-1]))

        # Make sure the graph only shows values with a positive x and y
        self.ax[0].set_xlim(xmin=0)
        self.ax[0].set_ylim(ymin=0)


    def _show_data_DGA(self, cur_gen, num_gens, pop_size, num_parents,
                       top_gen_score, top_score,
                       total_mean, gen_mean,
                       session_time_elapsed, gen_time_elapsed):
        '''Show the generation data.'''
        # Clear previous display
        self.ax[1].cla()

        # Turn off axes display
        self.ax[1].axis("off")

        # Set title
        self.ax[1].set_title("Session Data")

        # Display text
        self.ax[1].text(0.50, 0.50,
                f"Current Generation: {cur_gen} of {num_gens}\n" +
                f"Population Size: {pop_size}\n" +
                f"Parent Pool Size: {num_parents}\n\n" +

                f"Top Generation Score: {top_gen_score}\n" +
                f"Top Score: {top_score}\n\n" +

                f"Generation Mean: {gen_mean}\n" +
                f"Total Mean: {total_mean}\n\n" +

                f"Generation Time: {int(gen_time_elapsed//3600)}h {int(gen_time_elapsed//60 % 60)}m {int(gen_time_elapsed % 60)}s\n" +
                f"Session Time: {int(session_time_elapsed//3600)}h {int(session_time_elapsed//60 % 60)}m {int(session_time_elapsed % 60)}s\n",

                bbox={"facecolor": "white", "alpha": 1, "pad": 10},
                ha="center",
                va="center",
                size=12)
    

    def save_dga_session(self):
        ''' Save the data for the entire session. '''
        if not os_exists(r"./graphs"):
            os_makedirs(r"./graphs")
        if not os_exists(r"./graphs/DGA_session_graph.jpg"):
            plt_savefig(r"./graphs/DGA_session_graph.jpg")
