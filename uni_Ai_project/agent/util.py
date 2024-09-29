from referee.game import \
    PlayerColor, SpawnAction, SpreadAction, HexPos, HexDir
from .constants import MAX_TURNS, DIM, DIRS, MAX_SPAWN, MAX_TOT_POWER

def is_game_over(game_state, num_turns):
    if num_turns < 2: 
        return False
    
    color = PlayerColor.RED
    p1_power = 0 
    p2_power = 0
    for (color_cmp, power) in game_state.values():
        if color_cmp == color:
            p1_power += power 
        else: 
            p2_power += power 
    return any([
        num_turns >= MAX_TURNS,
        p1_power == 0,  
        p2_power == 0 
    ])


def update_board_spawn(game_state, color, x, y):
    game_state[(x,y)] = (color, 1) #(x,y): (colour, power)
    return game_state

def update_board_spread(game_state, color, x, y, x_dir, y_dir):
    (color_spreading, power) = game_state[(x, y)]
    game_state.pop((x,y))
    power_taken = 0
    steps = 1
    while steps <= power:
        if game_state.get(((x+x_dir*steps)%DIM, (y+y_dir*steps)%DIM)) is not None: # cell is occupied
            (color_captured, prev_power) = game_state[((x+x_dir*steps)%DIM, (y+y_dir*steps)%DIM)] # get previous power of the cell
            if color_spreading != color_captured:
                power_taken += (prev_power + 1)
            if prev_power < 6:
                game_state[((x+x_dir*steps)%DIM, (y+y_dir*steps)%DIM)] = (color, prev_power+1)
            else:
                game_state.pop(((x+x_dir*steps)%DIM, (y+y_dir*steps)%DIM))
        else: # cell is not occupied
            game_state[((x+x_dir*steps)%DIM, (y+y_dir*steps)%DIM)] = (color, 1) #(x,y): (colour, power)
        steps += 1
    return game_state, power_taken

def get_opposing_color(color):
    if color == PlayerColor.RED:
        return PlayerColor.BLUE
    else:
        return PlayerColor.RED

def format_action(action):
    if "spawn" in action:
        return SpawnAction(HexPos(action[1][0], action[1][1]))
    else:
        return SpreadAction(HexPos(action[1][0], action[1][1]), HexDir(action[2]))
    
def generate_next_moves(self, game_state, color):
    potential_next_moves = []
    for (x,y),(color_comp,_) in game_state.items(): # generate potential spread moves 
        for dir in DIRS:
            if color_comp == color:
                potential_next_moves.append(('spread', (x,y), (dir.r, dir.q)))
    if self.num_spawn <= MAX_SPAWN and self.power <= MAX_TOT_POWER:
        for x in range(0,7): # generate potential spawn moves
            for y in range(0,7):
                if (x,y) not in game_state.keys():
                    potential_next_moves.append(('spawn', (x,y)))
    return potential_next_moves

def generate_game_state(child, game_state, color):
    (x,y) = child[1]
    power_taken = None
    if 'spawn' in child:
        game_state = update_board_spawn(game_state, color, x, y)
    else:
        (x_dir, y_dir) = child[2]
        game_state, power_taken = update_board_spread(game_state, color, x, y, x_dir, y_dir)
    return game_state, power_taken

def can_infect(x1, y1, power1, x2, y2, power2):
    if (x1==x2):
        steps = 1
        while steps <= power1:
            if (y1+steps)%DIM == y2:
                return power2, (x1,y1), (0, 1)
            elif (y1-steps)%DIM == y2:
                return power2, (x1,y1), (0, -1)
            steps += 1
    elif (y1==y2):
        steps = 1
        while steps <= power1:
            if (x1+steps)%DIM == x2:
                return power2, (x1, y1), (1, 0)
            elif (x1-steps)%DIM == x2:
                return power2, (x1,y1), (-1, 0)
            steps += 1
    else:
        steps = 1
        while steps <= power1:
            if ((x1+steps)%DIM==x2 and (y1-steps)%DIM==y2):
                return power2, (x1, y1), (1, -1)
            elif ((x1-steps)%DIM==x2 and (y1+steps)%DIM ==y2):
                return power2, (x1, y1), (-1, 1)
            steps += 1
    return 0, None, None