
from typing import Any, Dict, Callable
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
from pathlib import Path

import sys
sys.path.append("../submission") 

from .environment import  SysadminEnv, CIRCLE, CROSS, EMPTY, get_opponent_policies
from .opponent_policies import *

# load opponent policies
opponent_policies = get_opponent_policies()

import json
from pathlib import Path

# Register environment
gym.register("Sysadmin-ED", partial(SysadminEnv))
env = gym.make("Sysadmin-ED")

def evaluate_policy(weights, test_episodes, agent_policy, opponent_policy_index):
    """
    Evaluates the agent's policy described by the learned weights by simulating the given number of episodes. 
    Returns the overall number of wins, draws, looses, and the statistical mean of the episode returns.
    """

    returns = []
    wins, draws, looses = 0,0,0

    for episode in range(test_episodes):
        state = env.reset()[0]
        env.set_opponent_policy(opponent_policies[opponent_policy_index])
        done = False
        
        while not done:
            
            action = agent_policy(state, weights)
            _, reward, done, _,_ = env.step(action)
            if done: 
                if reward == 1: 
                    wins = wins + 1
                elif reward == 0:
                    draws = draws + 1
                elif reward == -1:
                    looses = looses + 1

                returns.append(reward)  
          
    return wins, draws, looses, np.mean(returns)


def evaluate(training_algorithm_fn, agent_policy_fn):
    # Policy testing
    training_episodes = 5000 # Number of training episodes
    test_episodes = 100 
    test_runs = 5 # Number of test runs

    env = gym.make("Sysadmin-ED") 
    results = []

    for opponent_policy_index in range(4):
        successes = 0
        print(f"opponent policy": {opponent_policy_index})
        for test_run_index in range(test_runs):
            env = gym.make("Sysadmin-ED") 
            weights = training_algorithm_fn(training_episodes, opponent_policy_index) # learn the weights via function approximation learning
            
            # Check that number of episodes is not exceeded
            if env.get_reset_counter > training_episodes:
                    raise RuntimeError(f"Exceeded maximal number of calls of reset function")
            
            wins, draws, looses, average_return = evaluate_policy(weights, test_episodes, agent_policy_fn, opponent_policy_index) # evaluate the learned policy
            if average_return > 0.25:
                successes += 1
            print(f"Wins: {wins}, Draws: {draws}, Looses: {looses}, Average Return: {average_return}, Succeses: {successes}") # print results of the current test run
        if successes >= test_runs-1:
            beaten = True
        else:
            beaten = False
        print(f"Opponent policy {opponent_policy_index+1} beaten: {beaten}")

        # Append results
        results.append({
            "opponent_policy": opponent_policy_index+1,
            "beaten": beaten
        })
    return results