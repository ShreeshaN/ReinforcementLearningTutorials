# -*- coding: utf-8 -*-
"""
@created on: 9/8/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import numpy as np
from copy import deepcopy

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """ Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
"""
    value_function = np.zeros(nS)

    while True:
        change = 0
        for state_idx in range(nS):
            v = 0
            for action_idx, action_prob in enumerate(policy[state_idx]):  # for each state in nA
                for probability, nextstate, reward, terminal in P[state_idx][action_idx]:
                    v += action_prob * probability * (reward + gamma * value_function[nextstate])
            change = max(change, abs(v - value_function[state_idx]))
            value_function[state_idx] = v
        if change < tol:
            break
    return value_function


def calc_action_function(state_idx, value_function, nA, P, gamma):
    new_action = np.zeros(nA)
    for action_idx in range(nA):
        for probability, nextstate, reward, terminal in P[state_idx][action_idx]:
            new_action[action_idx] += probability * (reward + (gamma * value_function[nextstate]))
    return new_action


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
    for state_idx in range(nS):
        new_action_idx = np.argmax(calc_action_function(state_idx, value_from_policy, nA, P, gamma))
        new_policy[state_idx] = np.eye(nA)[new_action_idx]
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    while True:
        value_fn = policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8)
        improved_policy = policy_improvement(P, nS, nA, value_fn, gamma)
        if (policy == improved_policy).all():
            break
        policy = deepcopy(improved_policy)
    return policy, value_fn


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    while True:
        change = 0
        for state_idx in range(nS):
            sub_value_fn = V_new[state_idx]
            V_new[state_idx] = max(calc_action_function(state_idx, V_new, nA, P, gamma))
            change = max(change, abs(V_new[state_idx] - sub_value_fn))
        if change < tol:
            break
    improved_policy = policy_improvement(P, nS, nA, V_new, gamma)
    return improved_policy, V_new


def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game.
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0

    for _ in range(n_episodes):
        ob = env.reset()  # initialize the episode
        done = False
        while not done:
            if render:
                env.render()  # render the game
            action_id = np.argmax(policy[ob, :])
            ob, reward, done, info = env.step(action_id)
            total_rewards += reward
            if done:
                break
    return total_rewards
