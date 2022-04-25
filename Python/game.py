import os
import pygame
import random
import numpy as np
from agent import Agent
from genetics import GeneticAlgorithm
from helper import plot_data, Direction, Point, TILE_SIZE, FPS, WHITE, RED, BLACK

pygame.init()

# Fonts
try:
    FONT = pygame.font.Font("arial.ttf", int(TILE_SIZE*1.5))
except:
    FONT = pygame.font.SysFont("arial", int(TILE_SIZE*1.5))



class SnakeGameAI():
    ''' The base logic for the game itself. '''
    def __init__(self, tiles_wide=32, tiles_high=24, tiles_margin=4):
        self.width = tiles_wide*TILE_SIZE
        self.height = tiles_high*TILE_SIZE
        self.margin = tiles_margin*TILE_SIZE # Added to the bottom for diplaying data (score, episode, etc)

        # Initialze display
        self.true_display = pygame.display.set_mode((self.width, self.height + self.margin), pygame.RESIZABLE)
        self.false_display = self.true_display.copy()
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # Initialize game values
        self.reset()
        self.generation = 0
        self.agent_num = 0
        self.top_score = 0
        self.agent_episode = 0
        self.mean_score = 0.0

        # Reward values
        self.food_reward = 10
        self.death_reward = -10

        # Colors
        self.color1 = (0, 0, 0)
        self.color2 = (0, 0, 0)


    def reset(self):
        ''' Reset/Initialize base game state. '''
        # Default starting direction
        self.direction = Direction.UP

        # Set head, then add it to the snake, along with two
        # other body blocks
        self.head = Point(self.width//2, self.height//2)
        self.snake = [self.head,
                      Point(self.head.x, self.head.y+TILE_SIZE),
                      Point(self.head.x, self.head.y+(2*TILE_SIZE))]

        # Initialize score and food
        self.score = 0
        self._food_gen()
        self.frame_iteration = 0


    def _food_gen(self):
        ''' Randomly place food on the map. '''
        x = random.randint(0, (self.width-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        y = random.randint(0, (self.height-TILE_SIZE) // TILE_SIZE) * TILE_SIZE
        self.food = Point(x, y)

        # Check for conflicting values
        if self.food in self.snake:
            self._food_gen()


    def play_step(self, action):
        ''' Run a frame of the game. '''
        self.frame_iteration += 1
        reward = 0

        # Check for if the game has been closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move
        self._move(action) # Update the head
        self.snake.insert(0, self.head)
        
        # Check if game over
        # If the snake hasn't made enough progress, it's executed
        game_over = False
        if self.is_collision() or (self.frame_iteration > 125*len(self.snake)):
            game_over = True
            reward = self.death_reward
            return reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.food_reward
            self._food_gen()
        else:
            self.snake.pop()
        
        # Update ui and clock
        self._update_ui()
        self.clock.tick(FPS)

        # Return values for the agent to process
        return reward, game_over, self.score


    def is_collision(self, block=None):
        if block is None:
            block = self.head
        # Hits boundary
        if block.x > self.width - TILE_SIZE or block.x < 0 or block.y > self.height - TILE_SIZE or block.y < 0:
            return True
        # Hits itself
        if block in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        ''' Update the game screen. '''
        self.false_display.fill(BLACK)

        # Draw out the snake block by block
        for x, y in self.snake:
            pygame.draw.rect(self.false_display, self.color1, [x, y, TILE_SIZE, TILE_SIZE])
            pygame.draw.rect(self.false_display, self.color2, [x, y, TILE_SIZE, TILE_SIZE], 1)

        # Draw the food block
        pygame.draw.rect(self.false_display, RED, [self.food.x, self.food.y, TILE_SIZE, TILE_SIZE])

        # Draw a line for the margin
        pygame.draw.line(self.false_display, WHITE, (0, self.height), (self.width, self.height), width=2)

        # Show the current agent
        text = FONT.render(f"Agent {self.agent_num}", True, WHITE)
        self.false_display.blit(text, [0, int(self.height+(TILE_SIZE//4))])

        # Show the current episode
        text = FONT.render(f"Episode: {self.agent_episode}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+(TILE_SIZE//4))])

        # Show the current generation
        text = FONT.render(f"Generation: {self.generation}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+(TILE_SIZE//4))])

        # Show the current score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.false_display.blit(text, [0, int(self.height+((TILE_SIZE//4)+(2*TILE_SIZE)))])

        # Show the highest score
        text = FONT.render(f"Top Score: {self.top_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*9, int(self.height+((TILE_SIZE//4)+(2*TILE_SIZE)))])

        # Show the mean score
        text = FONT.render(f"Mean: {self.mean_score}", True, WHITE)
        self.false_display.blit(text, [TILE_SIZE*20, int(self.height+((TILE_SIZE//4)+(2*TILE_SIZE)))])

        # Update the display
        self.true_display.blit(pygame.transform.scale(self.false_display, self.true_display.get_rect().size), (0, 0))
        pygame.display.flip()


    def _move(self, action):
        '''
        Choose a new direction from straight, right, or left, where
        straight is to continue the current direction and right and
        left are to turn in either direction from the perspective
        of what direction the snake is currently heading.
        '''
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # No change (straight)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        # Right turn r -> d -> l -> u
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        # Left turn r -> u -> l -> d
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        # Set new direction to the class variable
        self.direction = new_dir

        # Update the head coordinates
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += TILE_SIZE
        elif self.direction == Direction.LEFT:
            x -= TILE_SIZE
        elif self.direction == Direction.DOWN:
            y += TILE_SIZE
        elif self.direction == Direction.UP:
            y -= TILE_SIZE
        self.head = Point(x, y)


def run(population_size=20, max_episodes=10, max_generations=10):
    agents = [Agent() for i in range(population_size)]
    game = SnakeGameAI()
    genetics = GeneticAlgorithm()

    scores, mean_scores = [], []
    for cur_gen in range(1, max_generations+1):
        game.generation = cur_gen

        for agent_num, agent in enumerate(agents):
            # Set colors
            game.color1 = agent.color1
            game.color2 = agent.color2
            
            # Set agent number
            game.agent_num = agent_num

            total_score = 0
            for cur_episode in range(1, max_episodes+1):
                agent.episode = cur_episode
                game.agent_episode = cur_episode
                run = True
                while run:
                    # Get old state
                    state_old = agent.get_state(game)

                    # Get move
                    final_move = agent.get_action(state_old)

                    # Perform move and get new state
                    reward, done, score = game.play_step(final_move)
                    state_new = agent.get_state(game)

                    # Train short memory
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)

                    # Remember
                    agent.remember(state_old, final_move, reward, state_new, done)

                    # Snake died
                    if done:
                        run = False
                        # Train long memory, plot result
                        game.reset()
                        agent.episode = cur_episode
                        game.agent_episode = cur_episode
                        agent.train_long_memory()

                        # Save model if it's the best (and update top score)
                        if score > game.top_score:
                            if not os.path.exists("./model"):
                                os.makedirs("./model")
                            agent.model.save(f"./models/model_gen{cur_gen}.h5")
                            game.top_score = score
                        
                        # Update agent's internal score if needed
                        if score > agent.top_score:
                            agent.top_score = score

                        total_score += score
                        game.mean_score = np.round((total_score / cur_episode), 3)
                        
                        # Record data
                        scores.append(score)
                        mean_scores.append(game.mean_score)
                        plot_data(scores, mean_scores, agent_num=game.agent_num, gen=cur_gen)
                        print(f"Agent {game.agent_num}")
                        print(f"Populatino {len(agents)}")
                        print(f"Episode: {cur_episode}")
                        print(f"Generation: {cur_gen}")
                        print(f"Score: {score}")
                        print(f"Top Score: {game.top_score}")
                        print(f"Mean: {game.mean_score}\n")

        # Make new population
        agents = genetics.breed_population(agents)



if __name__ == "__main__":
    run(population_size=20, max_episodes=5, max_generations=50)
