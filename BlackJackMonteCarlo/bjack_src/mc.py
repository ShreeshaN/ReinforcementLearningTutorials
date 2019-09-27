# -*- coding: utf-8 -*-
"""
@created on: 9/21/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import numpy as np
import random
from collections import defaultdict

"""     
    Monte-Carlo
    In this problem, we will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    We can test the correctness of our code 
    by typing 'nosetests -v mc_test.py' in the terminal.
"""


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    return 0 if observation[0] >= 20 else 1


def play_step(env, action_to_take):
    """
    Given the action to be taken, plays a step in the environment and returns the new set of values

    :param env: function
        OpenAI gym environment.
    :param action_to_take: int
        Action index to be taken for the current step.

    :return: next_state: 3-tuple
        (Player's sum, Dealer's sum, Boolean indicating if the player has ACE).
    :return: reward: int
        Reward received for choosing the given action
    :return: done: boolean
        Boolean indicating if the state is a terminal or not.
    """
    next_state, reward, done, info = env.step(action_to_take)
    return next_state, reward, done


def get_random_episode(env, policy):
    """
    Generates a list having episodes. Each episode in this list is generated until a terminal state is reached.
    :param env: function
        OpenAI gym environment.
    :param policy: function
        The policy  to be followed while choosing an action.
    :return: list
        List of generated episodes
    """
    new_set_of_episodes = []
    current_state = env.reset()
    while True:
        action_to_take = policy(current_state)
        next_state, reward, done = play_step(env, action_to_take)
        new_set_of_episodes.append((current_state, action_to_take, reward))
        if done:
            break
        current_state = next_state
    return new_set_of_episodes


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """
    Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    :param policy: function
        A function that maps an observation to action probabilities
    :param env: function
        OpenAI gym environment
    :param n_episodes: int
        Number of episodes to sample
    :param gamma: float
        Gamma discount factor
    :return V: defaultdict(float)
        A dictionary that maps from state to value
    """

    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    for i in range(n_episodes):
        new_set_of_episodes = get_random_episode(env, policy)
        states_set = set([episode[0] for episode in new_set_of_episodes])
        for i, state in enumerate(states_set):
            first_occurance = next(i for i, x in enumerate(new_set_of_episodes) if x[0] == state)
            total_reward = sum([(int(gamma) ** power) * episode[2] for power, episode in
                                enumerate(new_set_of_episodes[first_occurance:])])
            returns_sum[state] += total_reward
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]
    return V


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
    # A = np.ones(nA) * epsilon / float(nA)
    # best_action, prob_for_best_action = np.argmax(Q[state]), max(Q[state])
    # if prob_for_best_action > epsilon:
    #     A[best_action] += (1.0 - epsilon)
    #     return np.random.choice(np.arange(len(A)), p=A)
    # else:
    #     return np.random.choice(np.arange(len(A)))

    actions = np.ones(nA) * epsilon / float(nA)
    best_current_action = np.argmax(Q[state])
    actions[best_current_action] += (1.0 - epsilon)
    return np.random.choice(np.arange(len(actions)), p=actions)


def generate_random_episode_greedy(Q, nA, epsilon, env):
    """
    Generate episodes using epsilon greedy action chooser method.

    :param Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    :param nA: int
        Number of actions in the environment
    :param epsilon: float
        The probability to select a random action, range between 0 and 1
    :param env:  function
        OpenAI gym environment
    :return: list
        List of generated episodes
    """
    new_set_of_episodes = []
    current_state = env.reset()
    while True:
        action_to_take = epsilon_greedy(Q, current_state, nA, epsilon)
        next_state, reward, done = play_step(env, action_to_take)
        new_set_of_episodes.append((current_state, action_to_take, reward))
        if done:
            break
        current_state = next_state
    return new_set_of_episodes


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """
    Monte Carlo control with exploring starts.
    Find an optimal epsilon-greedy policy.

    :param env:  function
        OpenAI gym environment
    :param n_episodes:  int
        Number of episodes to sample
    :param gamma:  float
        Gamma discount factor
    :param epsilon:  float
        The probability to select a random action, range between 0 and 1
    :return: Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.

    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.

    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def decay(epsilon):
        return epsilon - (0.1 / n_episodes)

    for i in range(n_episodes):
        new_set_of_episodes = generate_random_episode_greedy(Q, env.action_space.n, epsilon, env)
        epsilon = decay(epsilon)
        for i in range(len(new_set_of_episodes)):
            state_action_pair = (new_set_of_episodes[i][0], new_set_of_episodes[i][1])
            first_occurance = next(
                    i for i, episode in enumerate(new_set_of_episodes) if (episode[0], episode[1]) == state_action_pair)
            g = sum([(int(gamma) ** power) * episode[2] for power, episode in
                     enumerate(new_set_of_episodes[first_occurance:])])
            returns_sum[state_action_pair] += g
            returns_count[state_action_pair] += 1
            Q[state_action_pair[0]][state_action_pair[1]] = returns_sum[state_action_pair] / returns_count[
                state_action_pair]
    return Q
