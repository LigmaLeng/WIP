import copy
from .util import get_opposing_color, generate_game_state, generate_next_moves, can_infect, is_game_over
from .constants import BIG_NUM, DIRS, DIM


def minimax(self, game_state, depth, alpha, beta, maximising):
    color = self._color if maximising == True else get_opposing_color(self._color)

    # evaluate the game_state of the previous move
    if depth == 0 or is_game_over(game_state, self.turn_count) == 1:
        #print(game_state)
        return eval(game_state, self._color), None 
    
    potential_moves = generate_next_moves(self, game_state, color)
    best_move = None
    if maximising:
        value = -BIG_NUM
        for child in potential_moves:
            tmp_game_state = copy.deepcopy(game_state)
            tmp_game_state, power_taken = generate_game_state(child, tmp_game_state, color)
            self.turn_count += 1
            if "spawn" in child:
                self.num_spawn += 1
                self.power += 1

            tmp, _ = minimax(self, tmp_game_state, depth-1, alpha, beta, False)
            self.turn_count -= 1
            if "spawn" in child:
                self.num_spawn -= 1
                self.power -= 1

            if tmp > value:
                value = tmp
                best_move = child
            if value >= beta: 
                break
            alpha = max(alpha, value)   
    else:
        value = BIG_NUM
        for child in potential_moves:
            tmp_game_state = copy.deepcopy(game_state)
            tmp_game_state, power_taken = generate_game_state(child, tmp_game_state, color)
            self.turn_count += 1
            if "spawn" in child:
                self.power += 1
            tmp, _ = minimax(self, tmp_game_state, depth-1, alpha, beta, True)
            self.turn_count -= 1
            if "spawn" in child:
                self.power -= 1
            if tmp < value:
                value = tmp
                best_move = child
            if value <= alpha:
                break
            beta = min(beta, value)
    return value, best_move

def eval(game_state, color):
    player_dict, player_num, player_power = {}, 0, 0 
    opp_dict, opp_num, opp_power = {}, 0, 0
    for (x,y), (color_cmp, power) in game_state.items():
        if color_cmp == color:
            player_dict[(x,y)] = (color_cmp, power)
            player_num += 1 
            player_power += power 
        else: 
            opp_dict[(x,y)] = (color_cmp, power)
            opp_num += 1 
            opp_power += power 
    
    h1 = huer_num_players(player_num, opp_power)
    h2 = heur_power(player_power, opp_power)
    h3 = heur_spreadability(player_dict, opp_dict)
    h4 = heur_proximity(player_dict, opp_dict) # a more dispersed players is susceptible. Decreasing the surface area exposed like an infection, will be more favourable. 
    return 1 * h1 + 5* h2 + 4* h3 + 1.5* h4

def huer_num_players(player_num, opp_num):
    return player_num - opp_num

def heur_power(player_power, opp_power):
    return player_power - opp_power

def heur_spreadability(player_dict, opp_dict):
    spreadability = find_spreadability(player_dict, opp_dict)
    catchability = find_spreadability(opp_dict, player_dict)
    return spreadability - catchability

def find_spreadability(player1, player2):
    power_attainable = 0
    for (x1,y1), (_, power1) in player1.items(): # player 1 = player
        for (x2, y2), (_, power2) in player2.items(): # player2 = opponent
            power, _, _ = can_infect(x1, y1, power1, x2, y2, power2)
            power_attainable += power
    return power_attainable


def heur_proximity(player_dict, opp_dict):
    # finds the number of available cells surrounding the player
    player_exposure = 0
    opp_exposure = 0
    for (x,y) in player_dict.keys():
        for dir in DIRS:
            if ((x+dir.r)%DIM, (y+dir.q)%DIM) not in player_dict:
                player_exposure += 1
    for (x,y) in opp_dict.keys():
        for dir in DIRS:
            if ((x+dir.r)%DIM, (y+dir.q)%DIM) not in player_dict:
                opp_exposure += 1
    return player_exposure - opp_exposure
