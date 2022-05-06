# SOURCE: https://python.plainenglish.io/d3qn-agent-with-prioritized-experience-replay-799f6e95264
import tensorflow as tf
from collections import deque
import random
import numpy as np
import math
import pandas as pd
import pickle
import os
import sys
from globals import path

from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_uniform, he_normal, Zeros
from tensorflow.keras.callbacks import History
from tensorflow import keras

import pygame

# pygame.init()

tf.keras.backend.clear_session()  # reset tensorflow session

# SOURCE: https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a
class D3QN(tf.keras.Model):
    def __init__(self, action_size):
        super(D3QN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(32, activation='relu', use_bias=True, kernel_initializer=he_uniform(), name="hidden1")
        self.hidden2 = tf.keras.layers.Dense(16, activation='relu', use_bias=True, kernel_initializer=he_uniform(), name="hidden2")
        # self.hidden3 = tf.keras.layers.Dense(16, activation='elu', kernel_initializer=he_normal(), name="hidden3")
        self.value_stream = tf.keras.layers.Dense(1, activation='linear')
        self.action_stream = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, input_data):
        x = self.hidden1(input_data)
        x = self.hidden2(x)
        # x = self.hidden3(x)
        value = self.value_stream(x)
        action = self.action_stream(x)
        Q_value = value + (action - tf.math.reduce_mean(action, axis=1, keepdims=True))
        return Q_value

    def advantage(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        # x = self.hidden3(x)
        action = self.action_stream(x)
        return action

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(32, activation='relu', use_bias=True, kernel_initializer=he_uniform(), name="hidden1")
        self.hidden2 = tf.keras.layers.Dense(16, activation='relu', use_bias=True, kernel_initializer=he_uniform(), name="hidden2")
        # self.hidden3 = tf.keras.layers.Dense(16, activation='elu', kernel_initializer=he_normal(), name="hidden3")
        self.q_value = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, input_data):
        x = self.hidden1(input_data)
        x = self.hidden2(x)
        # x = self.hidden3(x)
        Q_value = self.q_value(x)
        return Q_value

# DUELING DOUBLE DQN AGENT WITH PRIORITISED MEMORY
class D3QN_agent():
    def __init__(self, state_size, action_size, batch_size, double_ON, dueling_ON, prioritised_ON, softUpdates_ON):
        # initialise training parameters
        self.max_epsilon = 1  # max exploration probability
        self.min_epsilon = 0.01
        self.lambd = 0.00001  # exponential decay rate of epsilon between decisions (if close to 0 then nearly linear)

        self.gamma = 0.90  # discounting factor (if close to 1 then accounts for next state's Q value more when judging the current state-action pair)
        self.tau = 0.15  # inclusion rate of NN output to target network

        self.target_update_time = 60 * batch_size
        self.learning_rate = 0.00025  # learning rate for batch gradient descent optimisation of NN (larger value replaces past weights with batch-based parameters more)

        # initialise attributes
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.optimiser = Adam(learning_rate=self.learning_rate)

        # chose way of dealing with experiences and inputs
        self.double_ON = double_ON
        self.dueling_ON = dueling_ON
        self.prioritised_ON = prioritised_ON
        self.softUpdates_ON = softUpdates_ON

        if self.prioritised_ON:
            self.experience_replay = PrioritisedExperienceReplay(self.state_size)
            # importance sampling weights
            self.is_weight = np.power(self.batch_size, -self.experience_replay.bias_increase_per_sampling)
        else:
            self.experience_replay = ExperienceReplay(self.state_size)
            self.is_weight = 1/self.batch_size

        # initial epsilon-greedy behaviour
        self.epsilon = self.max_epsilon

        # build networks
        if self.dueling_ON:
            self.primary_network = D3QN(self.action_size)
            self.target_network = D3QN(self.action_size)
        else:
            self.primary_network = DQN(self.action_size)
            self.target_network = DQN(self.action_size)

        self.primary_network.compile(loss=self.PER_loss, optimizer=self.optimiser)
        self.target_network.compile(loss=self.PER_loss, optimizer=self.optimiser)

    # SOURCE: https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/03_priority_replay.py
    def PER_loss(self, y_target, y_pred):
        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))

    def update_epsilon(self, step):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lambd * step)

    # SOURCE: https://python.plainenglish.io/d3qn-agent-with-prioritized-experience-replay-799f6e95264
    def update_target_network(self):
        if self.softUpdates_ON:
            q_model_theta = self.primary_network.get_weights()
            target_model_theta = self.target_network.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.tau) + q_weight * self.tau
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_network.set_weights(target_model_theta)
        else:
            self.target_network.set_weights(self.primary_network.get_weights())

    def choose_action(self, state):
        # epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size - 1)
        else:
            if self.dueling_ON:
                q_values = self.primary_network.advantage(np.array([state]))
                # OLD VERSION (does not take advantage of dueling mechanism)
                # q_values = self.primary_network(state.reshape(1, -1))
            else:
                q_values = self.primary_network(np.array([state]))
            return np.argmax(q_values)

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_replay.add(state, action, reward, next_state, done)

    def train_network(self, batch_size):
        if self.prioritised_ON:
            tree_idx, batch, self.is_weight = self.experience_replay.get_batch(batch_size)
        else:
            batch = self.experience_replay.get_batch(batch_size)

        # SOURCE: https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/03_priority_replay.py
        states, actions, rewards, next_states, dones = self.experience_replay.extract_batch_arrays(batch)

        # predict Q(s,a) and Q(s',a') given the batch of past experienced states
        q_values_for_state = self.primary_network(states).numpy()
        q_values_for_next_state = self.primary_network(next_states).numpy()

        # Training primary network on batch but where Q values are from the target network
        ## preparation
        targets = q_values_for_state
        batch_idx = np.arange(batch_size)

        best_actions = np.argmax(q_values_for_next_state, axis=1)  # NOTE: Action is chosen from primary network but corresponding Q-value is from target network
        if self.double_ON:
            target_network_q_values_for_next_state = self.target_network(next_states)
            # Q-learning update rule but where the Q-values are drawn from the target network (DOUBLE Q-LEARNING)
            # updates[non_terminal_idx] += self.gamma * target_network_q_values_for_next_state.numpy()[batch_idx[non_terminal_idx], action[non_terminal_idx]]
            updates = rewards + self.gamma * target_network_q_values_for_next_state.numpy()[batch_idx, best_actions] * (1 - dones)
        else:
            # Vanilla Q-learning update rule
            # updates += self.gamma * q_values_for_next_state[batch_idx, action] * (1 - dones)
            updates = rewards + self.gamma * q_values_for_next_state[batch_idx, best_actions] * (1 - dones)

        targets[batch_idx, actions] = updates

        # update the experience priorities appearing in the SumTree based on the absolute descrepancy between old and new network output (before training)
        if self.prioritised_ON:
            abs_errors = np.abs(targets[batch_idx, actions] - q_values_for_state[batch_idx, actions])
            self.experience_replay.update_batch_priorities(tree_idx, abs_errors)

        # train primary network
        loss = self.primary_network.train_on_batch(states, targets)

        return loss

# Following class is used to link the agent with the environment
class AgentTrainer():
    def __init__(self, agent, environment, batch_size, screen, pygame_clock, fps, render_ON):
        self.agent = agent
        self.env = environment

        self.batch_size = batch_size
        self.pretrain_length = 3 * self.batch_size
        self.pretrained = False

        # for observation purposes
        self.screen = screen
        self.episode_render = render_ON
        self.clock = pygame_clock
        self.ticks = fps

        # TRIAL: Cut off learner after maximal number of steps; idea being that it
        # can retry early stages more (and get better at it) to better proceed in later
        # ones. If not cut off, it may learn a more careful behaviour even for early
        # stages (from the later stages where it struggles)
        self.max_steps = 10000

        # for saving the model and performance indicators
        self.model_save_time = 50

    def take_action(self, action):
        next_state, reward, done = self.env.step(action)
        next_state = next_state  # if not done else None
        return next_state, reward, done

    def print_epoch_summary(self, episode, total_epoch_reward, num_gates, highscore):
        print("Episode: %d - Reward: %.2f - Gates Cleared: %d - Highscore Gate Number: %d" % (episode, total_epoch_reward, num_gates, highscore))
        print("********************************************")

    # necessary to fill the memory at the start of the actual learning problem (deals with "empty memory problem")
    def pretrain(self):
        if self.pretrained:
            print("Already pretrained. Close window and call agent.train()")
            return True

        for i in range(self.pretrain_length):
            if i == 0:
                state = self.env.reset()

            # fill memory by randomly choosing actions
            action = np.random.choice(np.arange(self.agent.action_size))

            next_state, reward, done = self.take_action(action)

            # store experience gained
            self.agent.store_experience(state, action, reward, next_state, done)

            if done:
                # start new episode
                state = self.env.reset()
            else:
                # iterate without learning
                state = next_state

            if self.episode_render:
                self.env.render(self.screen)
                self.clock.tick(self.ticks)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        self.pretrained = True

        return self.pretrained

    def train(self, num_episodes):
        if not self.pretrained:
            print("Agent has not enough experience. Call agent.pretrain() first.")
            return

        total_timesteps = 0
        highest_number_gates_cleared = 0

        # for saving progress
        performance_file = path + '/saved_models/cache/performance_df.pkl'

        dict = {'episode': [], 'score': [], 'lifespan': [], 'no. gates': [], 'highest no. gates': [], 'average loss': [], 'terminal reason': []}

        average_loss_per_episode = []

        best_highscore_reward = -100
        # SOURCE: https://python.plainenglish.io/d3qn-agent-with-prioritized-experience-replay-799f6e95264
        for episode in range(0, num_episodes+1):
            # resetting the environment
            state = self.env.reset()

            # initialising important performance measures
            average_loss = 0
            total_epoch_reward = 0
            number_gates_cleared = 0

            done = False
            episode_steps = 0
            while not done:
                # choose action
                action = self.agent.choose_action(state)

                # perform action
                next_state, reward, done = self.take_action(action)
                if not done:
                    reward = reward
                else:
                    reward = -10
                    # next_state = None

                self.agent.store_experience(state, action, reward, next_state, done)

                # train agent on past experiences
                loss = self.agent.train_network(self.batch_size)

                # update epsilon for greedy strategy
                total_timesteps += 1
                self.agent.update_epsilon(total_timesteps)

                # update target network if target_update_time steps passed
                if self.agent.double_ON and (total_timesteps % self.agent.target_update_time == 0):
                    self.agent.update_target_network()

                # save performance indicators
                if reward > 0:
                    number_gates_cleared += 1
                average_loss += loss
                total_epoch_reward += reward

                if done or (episode_steps > self.max_steps):
                    done = True
                    if episode_steps > self.max_steps:
                        total_epoch_reward -= 10  # otherwise the early stop would indicate an increase in learning
                        print("Max steps reached. Break.")
                    average_loss /= total_epoch_reward
                    average_loss_per_episode.append(average_loss)
                    previous_highest_number_gates = highest_number_gates_cleared
                    highest_number_gates_cleared = max(highest_number_gates_cleared, number_gates_cleared)
                    if number_gates_cleared == highest_number_gates_cleared:
                        best_highscore_reward = max(total_epoch_reward, best_highscore_reward)
                    self.print_epoch_summary(episode+1, total_epoch_reward, number_gates_cleared, highest_number_gates_cleared)
                episode_steps += 1

                if self.episode_render:
                    self.env.render(self.screen)
                    # self.screen.fill((0, 0, 0, 0))
                    # pygame.display.update()
                    self.clock.tick(self.ticks)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # save in case of early abort
                        df = pd.DataFrame.from_dict(dict)
                        # pd.to_pickle(df, performance_file)
                        with open(performance_file, 'wb') as f:
                            pickle.dump(df, f)

                        pygame.quit()
                        sys.exit()

                # for clearing cache
                pygame.event.clear()
                    # pygame.quit()
                    # sys.exit()

                state = next_state

            # appending performance indicators every episode
            dict['episode'].append(episode)
            dict['score'].append(total_epoch_reward)
            dict['lifespan'].append(episode_steps)
            dict['no. gates'].append(number_gates_cleared)
            dict['highest no. gates'].append(highest_number_gates_cleared)
            dict['average loss'].append(average_loss)
            if episode_steps > self.max_steps:
                reason = 'max_steps'
            else:
                reason = 'terminal'
            dict['terminal reason'].append(reason)

            # saving model every model_save_time episodes or if it is the current front runner
            if (episode % self.model_save_time == 0):
                directory = path + "/saved_models/cache/models/episode_" + str(episode) + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.agent.primary_network.save_weights(directory + 'primary_network_weights', save_format='tf')
                self.agent.target_network.save_weights(directory + 'target_network_weights', save_format='tf')

                # save overall performance after training completed or when model is saved (to save something in case of freezes)
                df = pd.DataFrame.from_dict(dict)
                # pd.to_pickle(df, performance_file)
                with open(performance_file, 'wb') as f:
                    pickle.dump(df, f)
            elif (number_gates_cleared > max(15, previous_highest_number_gates)):
                directory = path + "/saved_models/cache/models/front_runners/highscore/episode_" + str(episode) + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.agent.primary_network.save_weights(directory + 'primary_network_weights', save_format='tf')
                self.agent.target_network.save_weights(directory + 'target_network_weights', save_format='tf')
            elif ((number_gates_cleared == max(15, highest_number_gates_cleared)) and (total_epoch_reward == best_highscore_reward)):
                directory = path + "/saved_models/cache/models/front_runners/time_improvement/episode_" + str(episode) + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.agent.primary_network.save_weights(directory + 'primary_network_weights', save_format='tf')
                self.agent.target_network.save_weights(directory + 'target_network_weights', save_format='tf')

        # save overall performance after training completed or when model is saved (to save something in case of freezes)
        df = pd.DataFrame.from_dict(dict)
        # pd.to_pickle(df, performance_file)
        with open(performance_file, 'wb') as f:
            pickle.dump(df, f)

        pygame.quit()
        sys.exit()

    def test(self, path_to_weights, epsilon_greedy_ON):
        # duplicate the agent's primary network and set the desired weights
        test_network = self.agent.primary_network
        test_network.load_weights(path_to_weights)

        # resetting the environment
        state = self.env.reset()

        # initialising important performance measures
        total_reward = 0
        number_gates_cleared = 0

        epsilon = 0.01
        done = False
        steps = 0
        while not done:
            # choose greedy action
            if epsilon_greedy_ON:
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.agent.action_size - 1)
                else:
                    if self.agent.dueling_ON:
                        q_values = test_network.advantage(np.array([state]))
                    else:
                        q_values = test_network(np.array([state]))

                    action = np.argmax(q_values)
            else:
                if self.agent.dueling_ON:
                    q_values = test_network.advantage(np.array([state]))
                else:
                    q_values = test_network(np.array([state]))

                action = np.argmax(q_values)

            # perform action
            next_state, reward, done = self.take_action(action)

            # save performance indicators
            if reward > 0:
                number_gates_cleared += 1
            total_reward += reward

            # check end of episode
            if done or (steps > self.max_steps):
                if number_gates_cleared > 0:
                    avg_steps_per_gate = steps/number_gates_cleared
                    avg_time_per_gate = avg_steps_per_gate/self.ticks
                else:
                    avg_time_per_gate = "nan"

                print("********************************************")
                if steps > self.max_steps:
                    total_reward -= 100  # otherwise the early stop would indicate an increase in learning
                    print("Max steps reached. Break.")
                else:
                    print("Dead.")
                if avg_time_per_gate == "nan":
                    print("Reward: %.2f - Gates Cleared: %d - Avg. time per gate: NaN sec." % (total_reward, number_gates_cleared))
                else:
                    print("Reward: %.2f - Gates Cleared: %d - Avg. time per gate: %.2f sec." % (total_reward, number_gates_cleared, avg_time_per_gate))
                print("********************************************")
                state = self.env.reset()

            steps += 1
            state = next_state

            # show screen
            self.env.render(self.screen)
            # self.clock.tick(self.ticks)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # for clearing cache
            pygame.event.clear()
                # pygame.quit()
                # sys.exit()

        return [total_reward, number_gates_cleared, avg_time_per_gate]

    def test_random_agent(self):
        # resetting the environment
        state = self.env.reset()

        # initialising important performance measures
        total_reward = 0
        number_gates_cleared = 0

        done = False
        steps = 0
        while not done:
            # choose random action
            action = np.random.randint(0, self.agent.action_size - 1)

            # perform action
            next_state, reward, done = self.take_action(action)

            # save performance indicators
            if reward > 0:
                number_gates_cleared += 1
            total_reward += reward

            # check end of episode
            if done or (steps > self.max_steps):
                if number_gates_cleared > 0:
                    avg_steps_per_gate = steps/number_gates_cleared
                    avg_time_per_gate = avg_steps_per_gate/self.ticks
                else:
                    avg_time_per_gate = "nan"

                print("********************************************")
                if steps > self.max_steps:
                    total_reward -= 100  # otherwise the early stop would indicate an increase in learning
                    print("Max steps reached. Break.")
                else:
                    print("Dead.")
                if avg_time_per_gate == "nan":
                    print("Reward: %.2f - Gates Cleared: %d - Avg. time per gate: NaN sec." % (total_reward, number_gates_cleared))
                else:
                    print("Reward: %.2f - Gates Cleared: %d - Avg. time per gate: %.2f sec." % (total_reward, number_gates_cleared, avg_time_per_gate))
                print("********************************************")
                state = self.env.reset()

            steps += 1
            state = next_state

            # show screen
            self.env.render(self.screen)
            # self.clock.tick(self.ticks)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # for clearing cache
            pygame.event.clear()
                # pygame.quit()
                # sys.exit()

        return [total_reward, number_gates_cleared, avg_time_per_gate]



# SOURCE: https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/03_priority_replay.py
# NOTE: Explanation about how it works: https://www.fcodelabs.com/2019/03/18/Sum-Tree-Introduction/
class SumTree(object):
    def __init__(self, capacity):
        # for creation of tree
        self.capacity = capacity  # number of leafs
        self.tree = np.zeros(2 * capacity - 1)  # total number of nodes in binary tree with self.capacity leafs
        # for retrival of data with specific priority
        self.data_pointer = 0
        self.data = np.empty(self.capacity, dtype=object)  # initialise tree with priority 0 in every leaf

    def add(self, priority, data):
        # starting with data_pointer = 0, the tree index is the left most leaf and every additional
        # value for data_pointer moves to the right along the leafs; i.e. we fill the leafs from
        # left to right with the data
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # only the leafs hold data of interest

        # for a given priority of the right-child leaf (say 3) and previously filled left-child leaf (say 5)
        # then the parent node has a priority defined as the sum of the two child priorities (i.e. 8).
        # This is iteratively updated along the levels of the tree up to the root in case a single leaf's
        # priority changes
        self.update(tree_index, priority)

        # next addition shall fill the next leaf
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]  # new priority score - previous priority score
        self.tree[tree_index] = priority
        # propagate the change up the tree to the root
        self.propagate(tree_index, change)

    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, value):
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            return idx
        if value <= self.tree[left_child_index]:
            return self.retrieve(left_child_index, value)
        else:
            return self.retrieve(right_child_index, value - self.tree[left_child_index])

    def get_leaf(self, value):
        idx = self.retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # easy way to extract total priority values in all leafs

class PrioritisedExperienceReplay:
    def __init__(self, state_size, capacity=100000):
        # hyperparameters for priority sampling
        self.const = 0.01  # minimal value of priority to avoid 0-probability of sampling
        self.priority_degree = 0.6  # if = 0 then we sample uniformly, if = 1 then only the highest priority is sampled every time
        self.high_priority_bias = 0.4  # starting value for bias of choosing samples that were chosen before

        self.bias_increase_per_sampling = 0.001  # incremental increase of self.high_priority_bias

        self.max_abs_error = 1  # maximal error in target difference

        # initialise SumTree
        self.tree = SumTree(capacity)

        # for extraction of batch values
        self.state_size = state_size

    def add(self, state, action, reward, next_state, terminated):
        # initially we do not know the priority of a new experience that was not yet selected for (batch) training
        # HENCE, we initialise it with the highest current priority to ensure that it will be chosen
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # in case the tree leafs are empty (e.g. before pre-training) we initialise it with "max_abs_error"
        if max_priority == 0:
            max_priority = self.max_abs_error

        data = [state, action, reward, next_state, terminated]
        self.tree.add(max_priority, data)

    # SOURCE: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    def get_batch(self, batch_size):
        # Following the idea of SumTree we divide range(0, total_priority) into batch_size segments and uniformly choose a value in each segment
        batch = []
        batch_idx, batch_priority = np.empty((batch_size, 1), dtype=np.int32), np.empty((batch_size, 1), dtype=np.float32)

        priority_grid_length = self.tree.total_priority / batch_size

        # increasing the "high priority bias" every time we sample a new batch
        self.high_priority_bias = np.min([1., self.high_priority_bias + self.bias_increase_per_sampling])  # max = 1

        # Calculating the max_weight
        index_of_first_data = self.tree.capacity - 1
        p_min = np.min(np.maximum(self.tree.tree[index_of_first_data:], self.const)) / self.tree.total_priority
        max_weight = np.power(p_min * batch_size, -self.high_priority_bias)

        for i in range(batch_size):
            # choose random number in current priority segment
            a, b = priority_grid_length * i, priority_grid_length * (i + 1)
            priority_val = np.random.uniform(a, b)

            # retrieve experience from tree that corresponds to this priority value
            index, priority, data = self.tree.get_leaf(priority_val)

            #P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  Importance sampling weights = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            batch_priority[i, 0] = np.power(batch_size * sampling_probabilities, -self.high_priority_bias)/ max_weight

            batch_idx[i] = index

            batch.append([data[0], data[1], data[2], data[3], data[4]])

        return batch_idx, batch, batch_priority

    def extract_batch_arrays(self, batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(self.state_size) if x[3] is None else x[3]) for x in batch])
        dones = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, dones

    def update_batch_priorities(self, tree_idx, abs_errors):
        # after an experience was chosen to (batch) update the DQN, the resulting target error is used to update their priority value in the tree
        abs_errors += self.const  # necessary to avoid 0-probability
        truncated_errors = np.minimum(abs_errors, self.max_abs_error)  # truncate the errors
        priority_values = np.power(truncated_errors, self.priority_degree)

        for index, priority in zip(tree_idx, priority_values):
            self.tree.update(index, priority)



class ExperienceReplay:
    def __init__(self, state_size, max_size=100000):
        self.buffer = deque(maxlen=max_size)
        self.state_size = state_size

    def add(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def extract_batch_arrays(self, batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(self.state_size) if x[3] is None else x[3]) for x in batch])
        dones = np.array([x[4] for x in batch])

        return states, actions, rewards, next_states, dones

    @property
    def buffer_size(self):
        return len(self.buffer)
