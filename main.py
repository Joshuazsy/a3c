"""
Implementation of Asynchronous Advantage Actor-Critic (A3C) control for the CartPole environment.
"""

from brain import Brain
from agent import Agent
from optimizers import Optimizer
import time

NUMBER_OF_AGENTS = 8
NUMBER_OF_OPTIMIZERS = 2
RUN_TIME_IN_MINUTES = 5

brain = Brain()
agents = [Agent(brain) for _ in range(NUMBER_OF_AGENTS)]
optimizers = [Optimizer(brain) for _ in range(NUMBER_OF_OPTIMIZERS)]

agents[0].render = True
for agent in agents:
    agent.run()
for optimizer in optimizers:
    optimizer.run()

time.sleep(RUN_TIME_IN_MINUTES*60)

for agent in agents:
    agent.stop()
for agent in agents:
    agent.join() # Let the agents finish their episode
for optimizer in optimizers:
    optimizer.stop()
for optimizer in optimizers:
    optimizer.join()

