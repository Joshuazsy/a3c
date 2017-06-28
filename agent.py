"""
Implementation of an asynchronous agent in the CartPole environment.
"""

import gym
import threading
import numpy as np
from collections import deque
import time


class Agent(threading.Thread):
    """
    An agent running in a Gym environment.

    Parameters
    ----------
    brain : Brain
        The brain to use and update.
    discount_factor : float, optional
        By how much should the agent discount its rewards over time? Also known as gamma. Default is 0.99.
    timesteps_until_update : integer, optional
        How many timesteps should we wait until updating? Default is 5.
    max_td_steps : integer, optional
        What is the maximum number of steps used in the temporal difference estimate of the advantage for every state?
        If larger than timesteps_until_update, will be capped to that value. Default is 5.
    """
    def __init__(self, brain, discount_factor=0.99, timesteps_until_update=5, max_td_steps=5):
        super().__init__()
        self.THREAD_DELAY = 0.001
        self.started = False
        self.render = False

        self.brain = brain
        self.discount_factor = discount_factor
        self.timesteps_until_update = timesteps_until_update
        self.max_td_steps = max_td_steps
        self.environment = gym.make("CartPole-v0")
        self.memory = deque()
        self.epsilon = 1.0

    def run(self):
        """
        Start the agent. It will start running episodes until explicitly stopped.
        """
        self.started = True
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
        state = self.environment.reset()
        timesteps = 0
        while True:
            time.sleep(self.THREAD_DELAY)

            if self.render:
                self.environment.render()
            action = self.brain.select_action(state, epsilon=self.epsilon)
            next_state, reward, terminal, _ = self.environment.step(action)
            self.memory.append({'state': state, 'action': action, 'reward': reward})
            timesteps += 1
            self.epsilon = 0.05 + (self.epsilon - 0.05) * np.exp(-0.001)

            if terminal or timesteps % self.timesteps_until_update == 0:
                self.push_training_samples(use_entire_memory=terminal)

    def push_training_samples(self, use_entire_memory=False):
        """
        Compute and push training samples into the brain training queue.

        Parameters
        ----------
        use_entire_memory : boolean, optional
            Should the entire memory be used? If False, use only the states for which n-step returns can be computed,
            where n=self.max_td_steps. If True, states for which less than n-step returns can be computed will also be
            included, with the largest number of steps possible. Default is False.
        """
        while self.memory:
            return_ = 0.0
            td_steps = max(self.max_td_steps, len(self.memory))
            for k in range(td_steps):
                return_ += self.discount_factor**k * self.memory[k]['reward']
                return_ += self.discount_factor**td_steps * self.brain.compute_value(self.memory[td_steps]['state'])
            self.brain.save_step(self.memory[0]['state'], self.memory[0]['action'], return_)
            self.memory.popleft()
            if not use_entire_memory and len(self.memory) < self.max_td_steps:
                break



