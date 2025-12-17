# Necessary Imports

from typing import Any, Dict, Callable
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
from pathlib import Path

import sys
sys.path.append("/submission") 
from agent import *  
from environment import SysadminEnv, CIRCLE, CROSS, EMPTY, get_opponent_policies


# load opponent policies
opponent_policies = get_opponent_policies()

import json
from pathlib import Path

# Register environment
gym.register("Sysadmin-ED", partial(SysadminEnv))
env = gym.make("Sysadmin-ED")

def evaluate_policy(weights):
    """
    Evaluates the agent's policy described by the learned weights by simulating the given number of episodes. 
    Returns the overall number of wins, draws, looses, and the statistical mean of the episode returns.
    """

    returns = []
    wins, draws, looses = 0,0,0

    for episode in range(500):
        
        state = env.reset()[0]
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

def evaluate():
    # Policy testing
    training_episodes = 5000 # Number of training episodes
    test_runs = 5 # Number of test runs

    env = gym.make("Sysadmin-ED") 
    results = []

    for i in range(4):
        print(f"Opponent Policy: {i+1}")
        env.set_opponent_policy = opponent_policies[i]
        weights = training_algorithm(training_episodes)

        print(f"Test Run: {i}")
        env = gym.make("Sysadmin-ED") 
        
        wins, draws, looses, average_return = evaluate_policy(weights) # evaluate the learned policy
        print(f"Wins: {wins}, Draws: {draws}, Looses: {looses}, Average Return: {average_return}") # print results of the current test run

        # Append results
        results.append({
            "opponent_policy": i+1,
            "wins": wins,
            "draws": draws,
            "looses": looses,
            "average_return": average_return
        })

    script_dir = Path(__file__).parent          # folder where the script is
    file_path = script_dir.parent / 'out' / 'results.json'  # results.json inside 'out'
    file_path.parent.mkdir(parents=True, exist_ok=True)  # create 'out' folder if it doesn't exist

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate()