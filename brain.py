"""
Implementation of the shared brain.
"""

import numpy as np
import tensorflow as tf
import threading
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as keras_backend

from collections import deque

# Hardcoded for CartPole-v0
STATE_SPACE_SIZE = 4
ACTION_SPACE_SIZE = 2


class Brain:
    """
    A shared brain.
    """
    def __init__(self):
        self.CRITIC_LOSS_WEIGHT = 0.5
        self.ENTROPY_PENALTY_WEIGHT = 0.01
        self.LEARNING_RATE = 5e-3
        self.BATCH_SIZE = 32

        self.session = tf.Session()
        self.model = self._build_model()
        self.graph, self.state_placeholder, self.selected_action_placeholder, \
            self.return_placeholder, self.minimization_step = self._build_tf_graph()
        self.training_queue = deque()
        self.training_queue_lock = threading.Lock()

    def _build_model(self):
        """
        Build an actor-critic neural network model.

        Returns
        -------
        keras.models.Model
            The constructed model instance.
        """
        keras_backend.set_session(self.session)
        keras_backend.manual_variable_initialization(True)

        input_ = Input(shape=(None, STATE_SPACE_SIZE))
        layer = Dense(units=16, activation='relu')(input_)
        action_output = Dense(units=ACTION_SPACE_SIZE, activation='softmax')(layer)
        value_output = Dense(units=1, activation='linear')(layer)
        model = Model(inputs=[input], outputs=[action_output, value_output])
        model._make_predict_function()  # Must be called before any threads are created
        return model

    def _build_tf_graph(self):
        """
        Build the A3C loss and optimization steps as a Tensorflow graph, and return it.

        Returns
        -------
        graph
            The Tensorflow graph.
        state
            Tensorflow placeholder for the state.
        selected_action
            Tensorflow placeholder for the action that ended up being selected.
        return_
            Tensorflow placeholder for the n-step return that was computed by the agent
            based on the received reward after taking the selected action.
        minimization_step
            Tensorflow operation to run a single step of minimizing the total loss of the
            actor-critic model.
        """
        state = tf.placeholder(shape=(None, STATE_SPACE_SIZE), dtype=tf.float32, name='state')
        action_probabilities, value = self.model(state)

        selected_action = tf.placeholder(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32, name='selected_action')
        probability_of_selected_action = tf.reduce_sum(action_probabilities * selected_action, axis=1, keep_dims=True)

        return_ = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='return')
        advantage = return_ - value

        actor_loss = - tf.stop_gradient(advantage) * tf.log(probability_of_selected_action + 1e-10)
        critic_loss = tf.square(advantage)
        entropy_penalty = tf.reduce_sum(action_probabilities * tf.log(action_probabilities + 1e-10),
                                        axis=1, keep_dims=True)
        total_loss = tf.reduce_mean(actor_loss + self.CRITIC_LOSS_WEIGHT*critic_loss
                                    + self.ENTROPY_PENALTY_WEIGHT*entropy_penalty)
        minimization_step = tf.train.RMSPropOptimizer(self.LEARNING_RATE, decay=.99).minimize(total_loss)

        self.session.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        graph.finalize()  # To avoid future modifications

        return graph, state, selected_action, return_, minimization_step

    def run_optimization_step(self):
        """
        Run a single optimization step on the model.

        Returns
        -------
        boolean
            Whether we did actually run an optimization step (True), or if we didn't because not enough data
            was available in the training queue (False).
        """
        if len(self.training_queue) < self.BATCH_SIZE:
            return False
        with self.training_queue_lock:
            states, actions, returns = zip(*self.training_queue)
            self.training_queue.clear()
        states = np.vstack(states)
        actions = np.vstack(actions)
        returns = np.vstack(returns)

        self.session.run(self.minimization_step, feed_dict={self.state_placeholder: states,
                                                            self.selected_action_placeholder: actions,
                                                            self.return_placeholder: returns})
        return True

    def save_step(self, state, action, return_):
        """
        Save a step from an asynchronous agent in the brain's training queue.

        Parameters
        ----------
        state : np.array of shape (4,)
            The state at the start of the step.
        action : np.array of shape (2,)
            The action taken at the step.
        return_ : float
            The estimated return for the state-action pair.
        """
        self.training_queue.append((state, action, return_))

    def action_probabilities(self, state):
        """
        Return the brain's action probabilities when in a given state.

        Parameters
        ----------
        state
            The environment state for which to compute the action probabilities.

        Returns
        -------
        action
            The action probabilities.
        """
        with self.graph.as_default():
            action_probabilities, _ = self.model.predict(state)
            return action_probabilities

    def select_action(self, state, epsilon=0.0):
        """
        Randomly select an action, perhaps epsilon-greedily.

        Parameters
        ----------
        state
            The environment state for which to select an action.
        epsilon : float
            With probability epsilon, a random action will be chosen instead.

        Returns
        -------
        action : np.array
            The selected action, one-hot encoded.
        """
        action = np.zeros((ACTION_SPACE_SIZE,))
        if np.random.uniform() < epsilon:
            index = np.random.choice(range(ACTION_SPACE_SIZE))
        else:
            probabilities = self.action_probabilities(state)
            index = np.random.choice(range(ACTION_SPACE_SIZE), p=probabilities)
        action[index] = 1
        return action

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
            with self.graph.as_default():
                _, value = self.model.predict(state)
                return value

    def __del__(self):
        self.session.close()