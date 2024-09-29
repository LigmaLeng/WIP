from referee.game import PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir
import steph
from program import Horus, Agent
from constants import R, B, MAX_TURNS, MAX_SUM_PWR, ROLL_D6, DIRS
##export PYTHONPATH="$PYTHONPATH:/home/ligmaleng/NN/agent/"
##CUPY_ACCELERATORS=cub,cutensor python agent/gym.py
import cupy as np


R_SIGN = 1
B_SIGN = -1

N_ROUNDS = 1000


class George:
    def __init__(self):
        self.horus = Horus()
        self.turns = 0

    # if roll == 0:
        self.red, self.blue = Agent(R), steph.Agent(B)
        self.r_name, self.b_name = "L", "S"
        # else:
        #     self.blue, self.red = Agent(B), steph.Agent(R)
        #     self.b_name, self.r_name = "l", "s"
        self.agents = [(self.red, self.blue), (self.blue, self.red)]

        self.end = False
        self.r_sum = 0
        self.b_sum = 0
        self.playouts = 0
        self.wins = 0
        self.wins_ = []
        self.mse=[]
        self.grads=[]
        return

    def run(self):
        for i in range(N_ROUNDS):
            while not self.end:
                self.turns += 1
                if self.turns > MAX_TURNS:
                    diff = self.r_sum - self.b_sum
                    if diff > 1:
                        self.win(R)
                        self.wins += 1
                    elif diff < -1:
                        self.win(B)
                    else: self.wins_.append(0)
                    other.turn(agent._color, action)
                    agent.turn(agent._color, action)
                    break
                for agent, other in self.agents:
                    if self.end == True:
                        continue
                    action = agent.action()
                    self.turn(agent._color, action)
                    if self.end == True:
                        continue
                    other.turn(agent._color, action)
                    agent.turn(agent._color, action)

            mse, grads = self.red.eveAluate.batch_info()
            self.mse.append(mse)
            self.grads.append(grads)
            
            if i % 20 == 0:
                self.red.upload()
            if i % 3 == 0:
                print("updates: {}\nepsilon: {}\nbatch_mse: {}\nbatch_sum_grad: {}".format(self.red.updates, self.red.epsilon, mse, grads))

            self.playouts += 1        
            print("red score: {}\nblue score: {}\nTurn: {}\nlast action: {} by {}".format(
                self.r_sum, self.b_sum, self.turns, action, agent._color))
            self.reset()
            print("{} wins / {} playouts".format(self.wins, self.playouts))

        self.upload_errors()
        self.red.upload()
        return
    
    def upload_errors(self):
        self.red.upload_errors()
        self.red.eveAluate.upload_errors()
        np.savez("errors_per_playout", mse=np.asarray(self.mse), grad=np.asarray(self.grads))
        np.savez("results", results=np.asarray(self.wins_))

    def reset(self):
        self.turns = 0
        self.horus.reset()
        self.red.reset()
        self.blue.reset()
        self.end = False
        self.r_sum = 0
        self.b_sum = 0
        if self.playouts % 20 == 0 and self.playouts < 960:
            self.blue.max_spawn += 1
        return
    
    def turn(self, color: PlayerColor, action: Action):
        match action:
            case SpawnAction(cell):
                # print(f"Testing: {color} SPAWN at {cell}")
                if color == R:
                    if not self.validate_spawn(cell):
                        print(f"INVALID SPAWN BY {color} at {cell}")
                        self.win(B)
                        return
                    else: 
                        self.horus.spawn(cell, sign=1)
                        self.r_sum += 1
                else:
                    if not self.validate_spawn(cell):
                        print(f"INVALID SPAWN BY {color} at {cell}")
                        self.win(R)
                        return
                    else: 
                        self.horus.spawn(cell, sign=-1)
                        self.b_sum += 1
                pass
            case SpreadAction(cell, direction):
                # print(f"Testing: {color} SPREAD from {cell}, {direction}")
                r, q = cell
                if color == R:
                    if self.horus.view[r, q] < 1:
                        print(f"INVALID SPREAD BY {color} from {cell}, {direction}")
                        self.win(B)
                        return
                    else: 
                        self.horus.spread(cell, direction, sign=1)
                        self.r_sum = self.horus.sum_R()
                        self.b_sum = self.horus.sum_B()
                else:
                    if self.horus.view[r, q] > 1:
                        print(f"INVALID SPREAD BY {color} from {cell}, {direction}")
                        # self.blue.print_game_state()
                        self.win(R)
                        return
                    else: 
                        self.horus.spread(cell, direction, sign=-1)
                        self.r_sum = self.horus.sum_R()
                        self.b_sum = self.horus.sum_B()
                self.check_win_con()

        
    
    
    def printeresting(self):
        render_print = {}
        board = self.horus.scry()
        for i in range(7):
            for j in range (7):
                targ = board[i,j].item()
                if targ < 0 :
                    render_print[(i,j)] = (self.b_name, -int(targ))
                elif targ > 0:
                    render_print[(i,j)] = (self.r_name, int(targ))
        print(self.render_board(render_print, ansi=True))

    def validate_spawn(self, cell):
        r, q = cell
        if self.horus.view[r, q] != 0:
            return False
        if MAX_SUM_PWR <= (self.r_sum + self.b_sum):
            return False
        return True
    
    def win(self, color:PlayerColor):
        if color == R:
            self.wins_.append(1)
            self.blue.award(-1,end=True)
            self.red.award(1,end=True)
        else:
            self.wins_.append(-1)
            self.blue.award(1,end=True)
            self.red.award(-1,end=True) 
        self.end = True
        
    def check_win_con(self):
        if self.b_sum == 0:
            if self.r_sum == 0:
                self.wins_.append(0)
                self.blue.award(0,end=True)
                self.red.award(0,end=True)
            else: 
                self.win(R)
                self.wins += 1
            self.end = True
        elif self.r_sum == 0:
            self.win(B)
            self.end = True
        return

    def lend_a_hand(self, sign=0):
        for r in range(7):
            for q in range(7):
                unit = self.horus.view[r][q].item()
                if unit == 0:
                    return SpawnAction(HexPos(r, q))
                if unit < 0 and sign < 0:
                    return SpreadAction(HexPos(r, q), DIRS[ROLL_D6()])
                if unit > 0 and sign > 0:
                    return SpreadAction(HexPos(r, q), DIRS[ROLL_D6()])
        
    def apply_ansi(self, str, bold=True, color=None):
        bold_code = "\033[1m" if bold else ""
        color_code = ""
        if color == "L":
            if self.r_name == "L":
                color_code = "\033[31m"
            else: color_code = "\033[34m"
        if color == "S":
            if self.b_name == "S":
                color_code = "\033[34m"
            else: color_code = "\033[31m"
        return f"{bold_code}{color_code}{str}\033[0m"

    def render_board(self, board: dict[tuple, tuple], ansi=True) -> str:
        dim = 7
        output = ""
        for row in range(dim * 2 - 1):
            output += "    " * abs((dim - 1) - row)
            for col in range(dim - abs(row - (dim - 1))):
                # Map row, col to r, q
                r = max((dim - 1) - row, 0) + col
                q = max(row - (dim - 1), 0) + col
                if (r, q) in board:
                    color, power = board[(r, q)]
                    text = f"{color}{power}".center(4)
                    if ansi:
                        output += self.apply_ansi(text, color=color, bold=True)
                    else:
                        output += text
                else:
                    output += " .. "
                output += "    "
            output += "\n"
        return output


if __name__ == "__main__":
    trainer = George()
    trainer.run()

    
