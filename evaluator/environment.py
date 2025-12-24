# Necessary Imports

from typing import Any, Dict, Callable
from dataclasses import dataclass
from functools import cached_property, partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .opponent_policies import get_opponent_policies

import json
from pathlib import Path

opponent_policies = get_opponent_policies()

opponent_policy_dict = opponent_policies[0]  # CHANGE TO PLAY AGAINST A DIFFERENT POLICY

# Definitions of possible field values
CROSS, EMPTY, CIRCLE = 2, 1, 0  

def get_rows(state: list[list[int]]) -> [list[list[int]], list[list[int]], list[list[int]]]:
    """
    Helper function: Returns list of rows, list of columns, and list of diagonals
    
    """

    # Compute rows
    rows = state

    # Compute columns
    columns = []
    for j in range(3):
        column = []
        for i in range(3):
            column.append(state[i][j])
        columns.append(column)
    
    #Compute diagonals
    diagonal0 = []
    diagonal1 = []
    for i in range(3):
        diagonal0.append(state[i][i])
        diagonal1.append(state[2-i][i])
    
    # Return rows, columns, and diagonals. 
    return rows, columns, [diagonal0, diagonal1]

# Gymnasium environment for Tic-Tac-Toe
class SysadminEnv(gym.Env):

    def __init__(
        self
    ) -> None:
        
        super().__init__()
        self.opponent_policy_dict = opponent_policy_dict
        self.action_space = spaces.MultiDiscrete([3,3]) # Action space 
        self.observation_space = spaces.MultiDiscrete([[3,3,3],[3,3,3],[3,3,3]]) # State space
        self.reset_counter = 0


    @property
    def get_reset_counter(self):
        return self.reset_counter
    

    @property
    def occupied_fields(self) -> int | None:
        """
        Returns the number of occupied fields.

        """
        if not hasattr(self, "_state"):
            return None
        
        res = 0
        for l in self._state:
            for v in l:
                if v != EMPTY:
                    res = res + 1
            
        return res
    
    
    @property
    def game_finished(self) -> int | None:
        """
        Returns None if game is not finished.

        Returns 0 if circle wins.
        Returns 1 if it is a draw.
        Returns 2 if crosses wins. 

        """
        rows, columns, diagonals = get_rows(self._state)

        for l in rows + columns + diagonals:
            if all(v == CROSS for v in l):
                return 2
            if all(v == CIRCLE for v in l):
                return 0
            
        if self.occupied_fields == 9:
            return 1
        else:
            return None
        
    def set_opponent_policy(self, opponent_policy_dict):
        self.opponent_policy_dict = opponent_policy_dict
        
    
    def opponent_policy (self) -> [int,int]:
        """
            Takes random action from the list of moves of the opponent policy.
        """

        if not hasattr(self, "_state"):
            raise Exception("Unable to find opponent move in uninitialized environment.")
        
        opponent_action_list = self.opponent_policy_dict[self._state.__str__()]
        return opponent_action_list[np.random.choice(len(opponent_action_list))]

    
    def perform_move(self, move: [int, int], cross: bool):
        """
        Returns the number of occupied fields.

        """
        if not hasattr(self, "_state"):
            raise Exception("Unable to perform move in uninitialized environment.")

        if self._state[move[0]][move[1]] != EMPTY:
            raise Exception("Unable to perform move on occupied field.")
        
        if cross: 
            self._state[move[0]][move[1]] = 2
        else:
            self._state[move[0]][move[1]] = 0

        
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment to its initial state.
        
        """
        
        super().reset(seed=seed)

        # increment reset_counter
        self.reset_counter += 1

        # All fields are empty initially
        self._state = [[EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY]]

        # Random choice whether agent or opponent makes the first move. 
        # In case of opponent, first move of opponent is performed.
        if np.random.random() < 0.5:   
            self.perform_move(self.opponent_policy(), False)
        
        return self._state, dict()
   

    def step(self, action: [int,int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Performs a step in the environment given an action of the agent.

        Return: new_state, reward, done, truncated, information_dictionary (last two return values are irrelevant for our purposes)  

        """
        
        # Perform agent's move
        self.perform_move(action, True)

        # Check whether game is finished and compute the return
        finished = self.game_finished  
        if self.game_finished == 2:
            return self._state, 1, True, False, dict()
        if self.game_finished == 1:
            return self._state, 0, True, False, dict()


        # Perform opponent's move
        self.perform_move(self.opponent_policy(), False)

        # Check whether game is finished and compute the return
        finished = self.game_finished  
        if finished is None:
            return self._state, 0, False, False, dict()
        elif finished == 0:
            return self._state, -1, True, False, dict()
        elif finished == 1:
            return self._state, 0, True, False, dict()    


    def display(self):
        """
        Prints the current state of the field to the command line. 

        """

        if not hasattr(self, "_state"):
           raise Exception("Unable to visualize uninitialized environment.")
       
        res = [["","",""],["","",""],["","",""]] 

        for i in range(3):
            for j in range(3):
                v = self._state[i][j]
                if v == CROSS:
                    res[i][j] = "X"
                elif v == EMPTY:
                    res[i][j] = " "
                elif v == CIRCLE:
                    res[i][j] = "O"
                else: 
                    raise Exception("Invalid value in TicTacToe Field")

        for l in res: 
            print(l)
    
        print("\n")

