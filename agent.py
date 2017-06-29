"""
Implementation of an asynchronous agent in the CartPole environment.
"""

import gym
import threading
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

from brain import brain

timesteps = 0


class Agent(threading.Thread):
    """
    An agent running in a Gym environment.

    Parameters
    ----------
    discount_factor : float, optional
        By how much should the agent discount its rewards over time? Also known as gamma. Default is 0.99.
    timesteps_until_update : integer, optional
        How many timesteps should we wait until updating? Default is 8.
    max_td_steps : integer, optional
        What is the maximum number of steps used in the temporal difference estimate of the advantage for every state?
        If larger than timesteps_until_update, will be capped to that value. Default is 8.
    """
    def __init__(self, render=False, discount_factor=0.99, timesteps_until_update=8, max_td_steps=8):
        super().__init__()
        self.THREAD_DELAY = 0.001
        self.started = False
        self.render = render
        self.figure = None
        self.plot = None
        self.total_number_of_episodes = 0
        self.episode_rewards = []

        self.discount_factor = discount_factor
        self.timesteps_until_update = timesteps_until_update
        self.max_td_steps = max_td_steps
        self.environment = gym.make("CartPole-v0")
        self.memory = deque()
        self.internal_epsilon = 1.0

    def run(self):
        """
        Start the agent. It will start running episodes until explicitly stopped.
        """
        self.started = True
        if self.render:
            self.figure = plt.figure()
            self.plot = self.figure.add_subplot(1, 1, 1)
        while self.started:
            self.run_episode()

    def stop(self):
        """
        Stop the agent. It will finish its current episode run and turn idle.
        """
        self.started = False

    def run_episode(self):
        """
        Run the agent for a single episode, saving the samples in the shared brain.
        """
        global timesteps
        total_rewards = 0.0
        state = self.environment.reset()

        while True:
            time.sleep(self.THREAD_DELAY)
            if self.render:
                self.environment.render()

            action_index, action = brain.select_action(state, epsilon=self.epsilon)
            next_state, reward, terminal, _ = self.environment.step(action_index)
            if terminal:
                next_state = None
            self.push_training_sample(state, action, reward, next_state)

            state = next_state
            timesteps += 1
            total_rewards += reward

            if terminal:
                self.total_number_of_episodes += 1
                if self.render:
                    # print("End of episode {}, total reward: {}"
                    #       .format(self.total_number_of_episodes, total_rewards))
                    self.episode_rewards.append(total_rewards)
                    self.plot_episode()
                break

    def push_training_sample(self, state, action, reward, next_state):
        """
        Compute and push a training sample into the brain training queue.
        """
        self.memory.append((state, action, reward, next_state))
        if next_state is None:
            # We have reached the end of the episode, flush the memory
            while self.memory:
                state, action, _, _ = self.memory[0]
                total_discounted_reward, n_discount, n_state = self.compute_return(self.memory)
                brain.save_step(state, action, total_discounted_reward, n_discount, n_state)
                self.memory.popleft()
        else:
            if len(self.memory) >= self.max_td_steps:
                # We can compute at least one full n-step return
                state, action, _, _ = self.memory[0]
                total_discounted_reward, n_discount, n_state = \
                    self.compute_return([self.memory[i] for i in range(self.max_td_steps)])
                brain.save_step(state, action, total_discounted_reward, n_discount, n_state)
                self.memory.popleft()

    def compute_return(self, trajectory):
        """
        Compute return components over a trajectory.

        Parameters
        ----------
        trajectory : list or deque of (np.array((4,0)), np.array((2,0)), float, np.array((4,0))
            The trajectory over which to compute the return components.

        Returns
        -------
        total_discounted_reward
            The total discounted reward for the trajectory.
        n_discount : float
            The discount factor for the n-step state value.
        n_state : np.array of size (4, 0)
            The state at the end of the trajectory.
        """
        total_discounted_reward = 0.0
        for k in range(len(trajectory)):
            total_discounted_reward += self.discount_factor**k * trajectory[k][2]
        n_discount = self.discount_factor**len(trajectory)
        n_state = trajectory[-1][3]
        return total_discounted_reward, n_discount, n_state

    @property
    def epsilon(self):
        """
        Compute the internal epsilon according to the schedule.
        """
        return 0.05 + 0.95 * np.exp(-1e-5 * timesteps)
    #     if timesteps >= 75000:
    #         return 0.15
    #     else:
    #         return 0.4 + timesteps * (0.4 - 0.15) / 75000

    def plot_episode(self):
        self.plot.clear()
        self.plot.set_title('Total rewards of each episode')
        self.plot.set_xlabel('Episode')
        self.plot.set_ylabel('Total reward')
        self.plot.plot(range(len(self.episode_rewards)), self.episode_rewards, '-')
        plt.pause(0.001)
