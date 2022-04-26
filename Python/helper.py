from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import display
import os


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



plt.ion() # turn on interactable plots
fig, ax = plt.subplots(2, 2) # initialize 2x2 graphs

def plot_data(all_scores, all_mean_scores, gen_scores, gen_mean_scores, gen_num, agent_scores, agent_mean_scores, agent_num):
    ''' Plot the data for the game. '''
    display.clear_output(wait=True)
    display.display(plt.gcf())
    #clearConsole()
    plt.clf()

    plot_aggregate(all_scores, all_mean_scores)
    plot_generation(gen_scores, gen_mean_scores, gen_num)
    plot_agent(agent_scores, agent_mean_scores, agent_num)

    plt.show(block=False)
    plt.pause(0.001)



def plot_aggregate(all_scores, all_mean_scores):
    ''' Plot all data collected by the game. '''
    ax[0, 0].title("Total Aggregate Data")
    ax[0, 0].xlabel("Number of Games")
    ax[0, 0].ylabel("Score")

    ax[0, 0].plot(all_scores, label="Scores")
    ax[0, 0].plot(all_mean_scores, label="Mean Scores")
    ax[0, 0].legend(loc="upper left")

    ax[0, 0].text(len(all_scores)-1, all_scores[-1], str(all_scores[-1]))
    ax[0, 0].text(len(all_mean_scores)-1, all_mean_scores[-1], str(all_mean_scores[-1]))

    ax[0, 0].xlim(xmin=0)
    ax[0, 0].ylim(ymin=0)


def plot_generation(gen_scores, gen_mean_scores, gen_num):
    ''' Plot the data for the current generation. '''
    ax[0, 1].title(f"Generation {gen_num} Data")
    ax[0, 1].xlabel("Number of Games")
    ax[0, 1].ylabel("Score")

    ax[0, 1].plot(gen_scores, label="Scores")
    ax[0, 1].plot(gen_mean_scores, label="Mean Scores")
    ax[0, 1].legend(loc="upper left")

    ax[0, 1].text(len(gen_scores)-1, gen_scores[-1], str(gen_scores[-1]))
    ax[0, 1].text(len(gen_mean_scores)-1, gen_mean_scores[-1], str(gen_mean_scores[-1]))

    ax[0, 1].xlim(xmin=0)
    ax[0, 1].ylim(ymin=0)


def plot_agent(agent_scores, agent_mean_scores, agent_num):
    ''' Plot the data for the current agent. '''
    ax[1, 0].title(f"Agent {agent_num} Data")
    ax[1, 0].xlabel("Number of Games")
    ax[1, 0].ylabel("Score")

    ax[1, 0].plot(agent_scores, label="Scores")
    ax[1, 0].plot(agent_mean_scores, label="Mean Scores")
    ax[1, 0].legend(loc="upper left")

    ax[1, 0].text(len(agent_scores)-1, agent_scores[-1], str(agent_scores[-1]))
    ax[1, 0].text(len(agent_mean_scores)-1, agent_mean_scores[-1], str(agent_mean_scores[-1]))

    ax[1, 0].xlim(xmin=0)
    ax[1, 0].ylim(ymin=0)


def save_graph(generation):
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    plt.savefig(f"./graphs/graph_gen{generation}.jpg")
