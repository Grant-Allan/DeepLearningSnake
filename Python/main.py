from run import RunGame
from pygame import quit as pyg_quit


if __name__ == "__main__":
    # Create RunGame object
    run_game = RunGame()

    run_game.run_human(fps=10)
    #run_game.run_dqn(fps=100, max_episodes=100)
    #run_game.run_grl(fps=100, population_size=50, max_episodes=20, max_generations=100)

    # Quit out of the pygame that was started originally
    pyg_quit()