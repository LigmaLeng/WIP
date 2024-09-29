from referee.game import PlayerColor, HexDir, HexPos
import random

DIRS = [HexDir.DownRight, HexDir.Down, HexDir.DownLeft, HexDir.UpLeft, HexDir.Up, HexDir.UpRight]
DIM = 7
MAX_TURNS = 343
MAX_SUM_PWR = 49
R = PlayerColor.RED
B = PlayerColor.BLUE
MAX_POW = 6
N_ACTIONS = 7 * 7 * 7
BIG_NUM = 999_999
MAX_SPAWN = 4 # can take any value <=49
MAX_TOT_POWER = 49
ROLL_FLOAT = lambda : random.random()
ROLL_POS = lambda : HexPos(ROLL_D7(), ROLL_D7())
ROLL_D6 = lambda : random.randint(0,5)
ROLL_D7 = lambda : random.randint(0,6)