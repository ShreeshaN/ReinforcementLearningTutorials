# -*- coding: utf-8 -*-
"""
@created on: 9/22/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import numpy as np
from collections import defaultdict

# -------------------------------------------------------------------------
"""
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of this code 
    by typing 'nosetests -v td_test.py' in the terminal.
"""


# -------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """
    A epsilon-greedy method to generate random action based on Q state
    :param Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    :param state: int
        current state
    :param nA: int
        Number of actions in the environment
    :param epsilon: float
        The probability to select a random action, range between 0 and 1
    :return: int
        action based current state

    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    A = np.ones(nA) * epsilon / float(nA)
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return np.random.choice(np.arange(len(A)), p=A)


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """
    On-policy TD control. Find an optimal epsilon-greedy policy.
    :param env: function
        OpenAI gym environment
    :param n_episodes: int
        Number of episodes to sample
    :param gamma: float
        Gamma discount factor, range between 0 and 1
    :param alpha: float
        step size, range between 0 and 1
    :param epsilon: float
        The probability to select a random action, range between 0 and 1
    :return: Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.

    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.

    """

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))

    def decay(epsilon):
        return 0.99 * epsilon

    for i in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, nA, epsilon)
        epsilon = decay(epsilon)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, nA, epsilon)
            td_target = reward + (gamma * Q[next_state][next_action])
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_error
            if done:
                break
            action = next_action
            state = next_state
    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """
    Off-policy TD control. Find an optimal epsilon-greedy policy.
    :param env: function
        OpenAI gym environment
    :param n_episodes: int
        Number of episodes to sample
    :param gamma: float
        Gamma discount factor, range between 0 and 1
    :param alpha: float
        step size, range between 0 and 1
    :param epsilon: float
        The probability to select a random action, range between 0 and 1
    :return: Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    """

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    for i in range(n_episodes):
        state = env.reset()

        while True:
            action = epsilon_greedy(Q, state, nA, epsilon)
            next_state, reward, done, info = env.step(action)
            next_best_action = np.argmax(Q[next_state])
            td_target = reward + (gamma * Q[next_state][next_best_action])
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_error
            if done:
                break
            state = next_state
    return Q
