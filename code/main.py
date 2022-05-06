from agent_env import *
from learning_method import *
import pygame
from pygame.locals import *
from globals import screen_size, screen_scaling
import numpy as np
import sys

"""
CHOOSE MODEL
"""
double_network = True
dueling_network = False

"""
CHOOSE LEARNING METHODOLOGY
"""
prioritised_memory = True
soft_target_updates = False
batch_size = 64
num_training_eps = 2000

"""
RENDERING DURING TRAINING
"""
render_ON = True


class Window():
    def __init__(self):
        pygame.init()
        # set up window
        flags = DOUBLEBUF
        bpp = 8
        self.screen = pygame.display.set_mode((screen_size[0], screen_size[1]), flags, bpp, vsync=1)
        self.clock = pygame.time.Clock()
        self.exit = False
        pygame.display.set_caption("DQN Algorithm")

        # initialise the environment
        self.ticks = 30
        self.env = Env(self.ticks)
        self.env.comp_vision_ON = False
        self.env.beam_ON = False

        # Initialise agent and agent trainer based on chosen model
        state_size = self.env.state_size
        action_size = self.env.action_size
        self.agent = D3QN_agent(state_size, action_size, batch_size, double_network, dueling_network, prioritised_memory, soft_target_updates)
        # agent = DDQN_agent(state_size, action_size)
        # agent = DQN_agent(state_size, action_size)
        self.agentTrainer = AgentTrainer(self.agent, self.env, batch_size, self.screen, self.clock, self.ticks, render_ON)

    def user_baseline_run(self):
        episode_steps = []
        no_gates_per_run = []
        steps = 0
        no_gates = 0
        while not self.exit:
            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                    # CURRENT FIX to obtain position of next gate (by hand)
                    # print(self.env.car.next_gate_pos)

            # DEBUGGING: Simulate DQN inputs from user
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_UP] & pressed[pygame.K_LEFT]:
                action_idx = 0
            elif pressed[pygame.K_UP] & pressed[pygame.K_RIGHT]:
                action_idx = 2
            elif pressed[pygame.K_DOWN] & pressed[pygame.K_LEFT]:
                action_idx = 6
            elif pressed[pygame.K_DOWN] & pressed[pygame.K_RIGHT]:
                action_idx = 8
            elif pressed[pygame.K_DOWN]:
                action_idx = 7
            elif pressed[pygame.K_UP]:
                action_idx = 1
            elif pressed[pygame.K_LEFT]:
                action_idx = 3
            elif pressed[pygame.K_RIGHT]:
                action_idx = 5
            else:
                action_idx = 4

            # Logic without learning
            next_state, reward, done = self.env.step(action_idx)

            # add performance indicators
            steps += 1
            if reward > 0:
                no_gates += 1

            if done:
                # append performance lists
                episode_steps.append(steps)
                no_gates_per_run.append(no_gates)

                # reset
                steps = 0
                no_gates = 0
                self.env.reset()

            # draw
            self.env.render(self.screen)

            #pygame.time.delay(50)
            self.clock.tick(self.ticks)

        pygame.quit()

        # print out performance
        print("Baseline created based on %d runs:" % len(no_gates_per_run))
        max_gates = np.max(no_gates_per_run)
        avg_gates = np.round(np.mean(no_gates_per_run))
        avg_steps = np.mean(episode_steps)
        if avg_gates > 0:
            avg_steps_per_gate = avg_steps/avg_gates
            avg_time_per_gate = np.round(avg_steps_per_gate/self.ticks, 2)
            print("Max. no. gates: %d - Avg. no gates: %d - Avg. time per gate: %.2f sec." % (max_gates, avg_gates, avg_time_per_gate))
        else:
            avg_time_per_gate = "NaN"
            print("Max. no. gates: %d - Avg. no gates: %d - Avg. time per gate: NaN sec." % (max_gates, avg_gates))

        # exit
        sys.exit()

    def train_run(self, num_training_eps):
        while not self.exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            train_OFF = self.agentTrainer.pretrain()

            if train_OFF and render_ON:
                white = (255, 255, 255)
                message = "Pre-training complete."
                message2 = "Closing window automatically."
                font = pygame.font.Font(None, 40)
                screen_message = font.render(message, True, white, False)
                screen_message2 = font.render(message2, True, white, False)
                center = (screen_size[0]/2 * screen_scaling, screen_size[1]/2 * screen_scaling)
                self.screen.fill((0, 0, 0, 0))
                self.screen.blit(screen_message, center)
                self.screen.blit(screen_message2, (center[0], center[1]+30))
                pygame.display.update()
                pygame.time.delay(5000)

            self.agentTrainer.train(num_training_eps)

        pygame.quit()
        sys.exit()

    def test_run(self, dir_to_weights, text_to_display, epsilon_greedy_ON):
        path_to_weights = dir_to_weights + "primary_network_weights"
        exit = False
        while not exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
            # create message at beginning
            white = (255, 255, 255)
            center = (screen_size[0]/2 * screen_scaling, screen_size[1]/2 * screen_scaling)
            font = pygame.font.Font(None, 40)

            message = text_to_display
            screen_message = font.render(message, True, white, False)
            self.screen.fill((0, 0, 0, 0))
            self.screen.blit(screen_message, center)
            pygame.display.update()
            pygame.time.delay(5000)

            total_reward, number_gates_cleared, avg_time_per_gate = self.agentTrainer.test(path_to_weights, epsilon_greedy_ON)

            # print score message
            pygame.time.delay(2000)
            score_message = "Total Reward:  " + str(np.round(total_reward, 2))
            gate_message =  "Gates Cleared: " + str(number_gates_cleared)
            score_message = font.render(score_message, True, white, False)
            gate_message = font.render(gate_message, True, white, False)
            self.screen.fill((0, 0, 0, 0))
            self.screen.blit(score_message, center)
            self.screen.blit(gate_message, (center[0], center[1]+40))
            if avg_time_per_gate != "nan":
                time_message = "Time per gate: " + str(np.round(avg_time_per_gate,2)) + " sec."
                time_message = font.render(time_message, True, white, False)
                self.screen.blit(time_message, (center[0], center[1]+80))
            pygame.display.update()
            pygame.time.delay(5000)

            pygame.quit()
            sys.exit()

    def test_run_random(self, text_to_display):
        exit = False
        while not exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
            # create message at beginning
            white = (255, 255, 255)
            center = (screen_size[0]/2 * screen_scaling, screen_size[1]/2 * screen_scaling)
            font = pygame.font.Font(None, 40)

            message = text_to_display
            screen_message = font.render(message, True, white, False)
            self.screen.fill((0, 0, 0, 0))
            self.screen.blit(screen_message, center)
            pygame.display.update()
            pygame.time.delay(5000)

            total_reward, number_gates_cleared, avg_time_per_gate = self.agentTrainer.test_random_agent()

            # print score message
            pygame.time.delay(2000)
            score_message = "Total Reward:  " + str(np.round(total_reward, 2))
            gate_message =  "Gates Cleared: " + str(number_gates_cleared)
            score_message = font.render(score_message, True, white, False)
            gate_message = font.render(gate_message, True, white, False)
            self.screen.fill((0, 0, 0, 0))
            self.screen.blit(score_message, center)
            self.screen.blit(gate_message, (center[0], center[1]+40))
            if avg_time_per_gate != "nan":
                time_message = "Time per gate: " + str(np.round(avg_time_per_gate,2)) + " sec."
                time_message = font.render(time_message, True, white, False)
                self.screen.blit(time_message, (center[0], center[1]+80))
            pygame.display.update()
            pygame.time.delay(5000)

            pygame.quit()
            sys.exit()


if __name__ == '__main__':
    screen = Window()

    """
    AGENT TRAINER (based on specific model before Window class)
    """
    # screen.train_run(num_training_eps)


    """
    TEST LEARNED WEIGHTS (MUST BE COMPATIBLE WITH CHOSEN MODEL)
    """
    dir_to_weights = "./saved_models/selection/32x16_double/episode_679/"
    epsilon_greedy_ON = False
    text_to_display = "32x16 Double Agent (Eps. 679)"
    screen.test_run(dir_to_weights, text_to_display, epsilon_greedy_ON)

    """
    CREATING USER BASELINE
    """
    # screen.user_baseline_run()
