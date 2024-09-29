# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent


from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexDir
from .minimax import minimax
from .random_agent import random_action
# from .power_greedy import power_greedy
from .util import format_action, update_board_spawn, update_board_spread
from .constants import BIG_NUM

# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """
        self._color = color
        self.game_state = {}
        self.num_spawn = 0
        self.turn_count = 0
        self.power = 0

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """
        match self._color:
            case PlayerColor.RED:
                if len(self.game_state) < 1: # the first move should always be a random spawn 
                    return random_action(self)
                
                ### FOR MINIMAX AGENT ####
                _, action = minimax(self, self.game_state, 2, -BIG_NUM, BIG_NUM, True)
                return format_action(action)
                
                #### FOR POWER GREEDY AGENT ####
                #return power_greedy(self)

                #### FOR RANDOM AGENT ####
                # return random_action(self) 
            case PlayerColor.BLUE:
                if len(self.game_state) < 2: # the first move should always be a random spawn 
                    return random_action(self)
                
                ### FOR MINIMAX AGENT ####
                _, action = minimax(self, self.game_state, 2, -BIG_NUM, BIG_NUM, True)
                return format_action(action)
                
                #### FOR POWER GREEDY AGENT ####
                # return power_greedy(self)

                #### FOR RANDOM AGENT ####
                # return random_action(self)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """

        match action:
            case SpawnAction(cell):
                self.game_state = update_board_spawn(self.game_state, color, cell.r, cell.q)
                self.turn_count += 1
                if color == self._color:
                    self.num_spawn += 1
                self.power += 1
                print(f"Testing: {color} SPAWN at {cell}")
                pass
            case SpreadAction(cell, direction):
                (color, power) = self.game_state[(cell.r, cell.q)]
                self.game_state, power_taken = update_board_spread(self.game_state, color, cell.r, cell.q, direction.r, direction.q)
                self.turn_count += 1
                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass
            
