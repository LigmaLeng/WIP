from referee.game import \
    SpawnAction, SpreadAction, HexPos, HexDir
import random
from .constants import MAX_SPAWN, MAX_TOT_POWER, DIRS

def random_action(self):
    # decides to spread or spawn
    move = ["spawn", "spread"]
    if len(self.game_state) <= 1: 
        action = "spawn"
    elif self.num_spawn < MAX_SPAWN and self.power <= MAX_TOT_POWER:
        action = random.choice(move)
    else:
        action = "spread"
    
    # randomly choose x and y, and direction
    if action == "spawn":
        valid = 0
        while valid == 0:
            x = random.randint(0,6)
            y = random.randint(0,6)
            restricted_spawns_moves = []
            if len(self.game_state) == 1:
                for (x_existing,y_exisiting) in self.game_state:
                    for dir in DIRS:
                        restricted_spawns_moves.append((x_existing+dir.r, y_exisiting+dir.q))
            if ((x,y) not in self.game_state) and ((x,y) not in restricted_spawns_moves):
                valid = 1
        return SpawnAction(HexPos(x, y))
    else:
        players = [(x,y) for ((x,y),(color, _)) in self.game_state.items() if color == self._color]
        (x,y) = random.choice(players)
        direction = random.choice(DIRS)
        return SpreadAction(HexPos(x, y), HexDir((direction.r, direction.q)))