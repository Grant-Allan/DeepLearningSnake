from run import Run
from pygame import quit as pyg_quit


if __name__ == "__main__":
    #Run.run_human(fps=30)
    #Run.run_dqn(fps=100, max_episodes=100)
    Run.run_grl(fps=100, population_size=50, max_episodes=20, max_generations=100)

    # Quit out of the pygame that was started originally
    pyg_quit()