"""
Implementation of the shared brain.
"""


class Brain:
    """
    A shared brain.
    """
    def __init__(self):
        pass

    def save_step(self, state, action, advantage):
        """
        Save a step from an asynchronous agent in the brain's training queue.
        """
        raise NotImplementedError

    def select_action(self, state):
        """
        Select an action when in a given state.

        Parameters
        ----------
        state
            The environment state for which to select an action.

        Returns
        -------
        action
            The selected action.
        """
        raise NotImplementedError

    def compute_value(self, state):
        """
        Compute the value of a state.

        Returns
        -------
        float
            The value of the state.
        """
        if state is None:
            return 0.0
        else:
            raise NotImplementedError
