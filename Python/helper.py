from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import display


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
FPS = 40

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



plt.ion()
def plot_data(scores, mean_scores, agent_num=0, gen=1):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    #clearConsole()
    plt.clf()

    plt.title(f"SCORE TRACKER\nGeneration {gen} Agent {agent_num}")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    plt.plot(scores, label="Scores")
    plt.plot(mean_scores, label="Mean Scores")
    plt.legend(loc="upper left")

    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.show(block=False)
    plt.pause(0.001)


