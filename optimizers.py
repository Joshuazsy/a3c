"""
Optimizers for the A3C algorithm.
"""

import threading
import time


class Optimizer(threading.Thread):
    """
    An independent unit in charge of optimizing the actor-critic model.

    Parameters
    ----------
    brain : Brain
        The brain to use and update.
    """
    def __init__(self, brain):
        super().__init__()
        self.started = False

        self.brain = brain

    def run(self):
        """
        Start the optimizer. It will start optimizing until explicitly stopped.
        """
        self.started = True
        while self.started:
            if not self.brain.run_optimization_step():
                # If list is empty, don't keep trying until timeout, just yield already
                time.sleep(0)

    def stop(self):
        """
        Stop the agent. It will stop trying to optimize the brain.
        """
        self.started = False
