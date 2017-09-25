"""
Implementation of Asynchronous Advantage Actor-Critic (A3C) control for the CartPole environment.
"""

from agent import Agent
from optimizers import Optimizer
import time

NUMBER_OF_AGENTS = 8
NUMBER_OF_OPTIMIZERS = 2
RUN_TIME_IN_MINUTES = 5

agents = [Agent() for _ in range(NUMBER_OF_AGENTS)]
optimizers = [Optimizer() for _ in range(NUMBER_OF_OPTIMIZERS)]

agents[0].render = True
for agent in agents:
    agent.start()
for optimizer in optimizers:
    optimizer.start()

try:
    time.sleep(RUN_TIME_IN_MINUTES*60)
except KeyboardInterrupt:
    for agent in agents:
        agent.stop()
    for agent in agents:
        agent.join()  # Let the agents finish their episode
    for optimizer in optimizers:
        optimizer.stop()
    for optimizer in optimizers:
        optimizer.join()

for agent in agents:
    agent.stop()
for agent in agents:
    agent.join()  # Let the agents finish their episode
for optimizer in optimizers:
    optimizer.stop()
for optimizer in optimizers:
    optimizer.join()

# brain.saver.save(brain.session, 'model/model')
