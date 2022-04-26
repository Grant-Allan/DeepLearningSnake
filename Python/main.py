from run import Run
from pygame import quit as pyg_quit

if __name__ == "__main__":
    #Run.run_human()
    #Run.run_dqn()
    Run.run_grl(population_size=50, max_episodes=20, max_generations=100)

    pyg_quit()