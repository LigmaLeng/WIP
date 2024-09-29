from referee.game import \
    SpawnAction, SpreadAction, HexPos, HexDir
import random
from .agent.util import can_infect
from .agent.constants import MAX_SPAWN, MAX_TOT_POWER, DIRS

def power_greedy(self):
    player_dict, opp_dict = {}, {}
    for (x,y), (color_cmp, power) in self.game_state.items():
        if color_cmp == self._color:
            player_dict[(x,y)] = (color_cmp, power)
        else: 
            opp_dict[(x,y)] = (color_cmp, power)

    max_power_attainable = 0
    
    for (x1,y1), (_, power1) in player_dict.items(): # player 1 = player
        for (x2, y2), (_, power2) in opp_dict.items(): # player2 = opponent
            power, pos, dir = can_infect(x1, y1, power1, x2, y2, power2)    
            if power > max_power_attainable:
                max_power_attainable = power
                x, y, x_dir, y_dir = pos[0], pos[1], dir[0], dir[1]

    if max_power_attainable > 0:
        return SpreadAction(HexPos(x, y), HexDir((x_dir, y_dir)))
    elif self.num_spawn < MAX_SPAWN and self.power <= MAX_TOT_POWER:
        valid = 0
        while valid == 0:
            x = random.randint(0,6)
            y = random.randint(0,6)
            if (x,y) not in self.game_state:
                valid = 1
        return SpawnAction(HexPos(x, y))
    else:
        (x,y) = random.choice(list(player_dict.keys()))
        direction = random.choice(DIRS)
        return SpreadAction(HexPos(x, y), HexDir((direction.r, direction.q)))
