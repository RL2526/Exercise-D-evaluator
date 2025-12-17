# Necessary Imports

from typing import Any, Dict, Callable
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Load opponent policy from .json-file. 

import json
from pathlib import Path

def get_opponent_policies():
    opponent_policy_file = Path('Opponent_Policies') # Change filename to play against different policy.

    script_dir = Path(__file__).parent
    policy_path = script_dir / 'Opponent_Policies' / 'policy1.json'
    if policy_path.exists():
        with open(policy_path) as f:
            opponent_policy_1 = json.load(f)

    policy_path = script_dir / 'Opponent_Policies' / 'policy2.json'
    if policy_path.exists():
        with open(policy_path) as f:
            opponent_policy_2 = json.load(f)

    policy_path = script_dir / 'Opponent_Policies' / 'policy3.json'
    if policy_path.exists():
        with open(policy_path) as f:
            opponent_policy_3 = json.load(f)

    policy_path = script_dir / 'Opponent_Policies' / 'policy4.json'
    if policy_path.exists():
        with open(policy_path) as f:
            opponent_policy_4 = json.load(f)

    opponent_policies = [opponent_policy_1, opponent_policy_2, opponent_policy_3, opponent_policy_4]
    return opponent_policies