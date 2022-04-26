from run import Run
from pygame import quit as pyg_quit


if __name__ == "__main__":
    runner = Run()

    #runner.run_human(fps=10)
    runner.run_dqn(fps=100, max_episodes=100)
    #runner.run_grl(fps=100, population_size=50, max_episodes=20, max_generations=100)

    # Quit out of the pygame that was started originally
    pyg_quit()