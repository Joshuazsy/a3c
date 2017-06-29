"""
Implementation of the shared brain.
"""

import numpy as np
import tensorflow as tf
import threading
from keras.layers import Dense

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
        self.saver = None
        self.graph, self.state_ph, self.selected_action_ph, self.return_ph, self.policy, \
            self.value, self.minimization_step, self.total_loss = self._build_tf_graph()
        self.training_queue = deque()
        self.training_queue_lock = threading.Lock()
        self.number_of_updates = 0

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
        policy
            Tensorflow tensor for the computed action probabilities (i.e. policy).
        value
            Tensorflow tensor for the computed value function.
        minimization_step
            Tensorflow operation to run a single step of minimizing the total loss of the
            actor-critic model.
        total_loss
            Tensorflow tensor for the compute total loss.
        """
        state = tf.placeholder(shape=(None, STATE_SPACE_SIZE), dtype=tf.float32, name='state')
        layer = Dense(units=16, activation='relu')(state)
        policy = Dense(units=ACTION_SPACE_SIZE, activation='softmax')(layer)
        value = Dense(units=1, activation='linear')(layer)

        selected_action = tf.placeholder(shape=(None, ACTION_SPACE_SIZE), dtype=tf.float32, name='selected_action')
        probability_of_selected_action = tf.reduce_sum(policy * selected_action, axis=1, keep_dims=True)

        return_ = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='return')
        advantage = return_ - value

        actor_loss = - tf.stop_gradient(advantage) * tf.log(probability_of_selected_action + 1e-10)
        critic_loss = tf.square(advantage)
        entropy_penalty = tf.reduce_sum(policy * tf.log(policy + 1e-10),
                                        axis=1, keep_dims=True)
        total_loss = tf.reduce_mean(actor_loss + self.CRITIC_LOSS_WEIGHT*critic_loss
                                    + self.ENTROPY_PENALTY_WEIGHT*entropy_penalty)
        minimization_step = tf.train.RMSPropOptimizer(self.LEARNING_RATE, decay=.99).minimize(total_loss)

        # self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        graph.finalize()  # To avoid future modifications

        return graph, state, selected_action, return_, policy, value, minimization_step, total_loss

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
            states, actions, total_discounted_rewards, n_discounts, n_states, \
                n_state_is_terminal_flags = zip(*self.training_queue)
            self.training_queue.clear()
        states = np.vstack(states)
        actions = np.vstack(actions)
        n_discounts = np.vstack(n_discounts)
        total_discounted_rewards = np.vstack(total_discounted_rewards)
        n_states = np.vstack(n_states)
        n_state_is_terminal_flags = np.vstack(n_state_is_terminal_flags)

        returns = total_discounted_rewards + n_discounts * self.compute_value(n_states) * n_state_is_terminal_flags

        _, total_loss = self.session.run([self.minimization_step, self.total_loss],
                                         feed_dict={self.state_ph: states,
                                                    self.selected_action_ph: actions,
                                                    self.return_ph: returns})
        self.number_of_updates += 1
        if self.number_of_updates % 100 == 0:
            print("Loss after {} updates: {}".format(self.number_of_updates, total_loss))
        return True

    def save_step(self, state, action, total_discounted_reward, n_discount, n_state):
        """
        Save a step from an asynchronous agent in the brain's training queue.

        Parameters
        ----------
        state : np.array of shape (STATE_SPACE_SIZE,)
            The state at the start of the step.
        action : np.array of shape (ACTION_SPACE_SIZE,)
            The action taken at the step.
        total_discounted_reward : float
            The estimated return for the state-action pair.
        n_discount : int
            Discount factor for the n-state value.
        n_state : np.array of shape (STATE_SPACE_SIZE,)
            The state in n steps.
        """
        if n_state is None:
            self.training_queue.append((state, action, total_discounted_reward, n_discount,
                                        np.zeros((STATE_SPACE_SIZE,)), 0.0))
        else:
            self.training_queue.append((state, action, total_discounted_reward, n_discount,
                                        n_state, 1.0))

    def compute_policy(self, state):
        """
        Return the brain's action probabilities when in a given state.

        Parameters
        ----------
        state : np.array of shape (batch_size, STATE_SPACE_SIZE)
            The environment state for which to compute the action probabilities.

        Returns
        -------
        policy : np.array of shape (batch_size, ACTION_SPACE_SIZE)
            The action probabilities.
        """
        with self.graph.as_default():
            policy = self.session.run(self.policy, feed_dict={self.state_ph: state})
            return policy

    def select_action(self, state, epsilon=0.0):
        """
        Randomly select an action, perhaps epsilon-greedily.

        Parameters
        ----------
        state : np.array of shape (STATE_SPACE_SIZE,)
            The environment state for which to select an action.
        epsilon : float
            With probability epsilon, a random action will be chosen instead.

        Returns
        -------
        index : integer
            The index of the action.
        action : np.array
            The selected action, one-hot encoded.
        """
        action = np.zeros((ACTION_SPACE_SIZE,))
        if np.random.uniform() < epsilon:
            index = np.random.choice(range(ACTION_SPACE_SIZE))
        else:
            probabilities = self.compute_policy(state.reshape(1, -1)).ravel()
            index = np.random.choice(range(ACTION_SPACE_SIZE), p=probabilities)
        action[index] = 1
        return index, action

    def compute_value(self, state):
        """
        Compute the value of a state.

        Parameters
        ----------
        state : np.array of shape (batch_size, STATE_SPACE_SIZE)
            The environment state for which to compute the value.

        Returns
        -------
        np.array of shape (batch_size,)
            The value of the state.
        """
        with self.graph.as_default():
            value = self.session.run(self.value, feed_dict={self.state_ph: state})
            return value

    def __del__(self):
        self.session.close()

# A global brain
brain = Brain()
