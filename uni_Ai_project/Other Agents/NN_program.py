# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir
from ..constants import DIM, ROLL_POS, ROLL_FLOAT, DIRS, ROLL_D6, MAX_POW
from .eden import Eve
import numpy as np
import random

N_ACTIONS = 7 * 7 * 7
ROLL_ACTION = lambda: random.randint(0, N_ACTIONS-1)




class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
    # def __init__(self, color: PlayerColor):
        """
        Initialise the agent.
        """
        self._color = color

        self.horus = Horus()
        self.toth = Toth(buffer_size=1000, phi_size=DIM)
        self.eveAluate = Eve(N_ACTIONS, alpha = 0.1)
        # self.eveAluate.download()
        self.naiEve = Eve(N_ACTIONS, alpha = 0.1)
        # self.naiEve.learn_from(self.eveAluate)

        self.actions = [i for i in range(N_ACTIONS)]

        self.epsilon, self.decay, self.decomposed , self.gamma = 1, 1e-3, 0.01, 0.99
        self.start, self.turns, self.end, self.reward= True, 0, False, 0
        self.minibatch_size, self.expulsion_period = 10, 200
        self.updates=0
        self.mse=[]
        self.grads=[]
        

    def reset(self):
        self.start, self.turns = True, 0
        self.award(-self.reward, end=True)
        self.horus.reset()
        return
    
    def award(self, reward:int, end=False):
        self.reward += reward
        self.end = end
        return


    def action(self, **referee: dict) -> Action:
    # def action(self) -> Action:
        """
        Return the next action to take.
        """
        if self.start == True:
            self.start = False
            return SpawnAction(ROLL_POS())
        elif self.epsilon > ROLL_FLOAT():
            # while True:
            r, q = ROLL_POS()
            if self.horus.view[r][q].item() == 0:
                return SpawnAction(HexPos(r, q))
            if self.horus.view[r][q].item() > 0:
                return SpreadAction(HexPos(r, q), DIRS[ROLL_D6()])
        else:
            actions = self.naiEve.advantage(self.horus.encode())
            actions = np.argsort(actions, axis=0)
            i = -1
            while True:
                i += 1
                r, q, move = self.index_actions(actions[i].item())
                sign = self.horus.view[r][q].item()
                if sign < 0:
                    continue
                if move == 0 and sign == 0:
                    return SpawnAction(HexPos(r, q))
                if move != 0 and sign > 0:
                    return SpreadAction(HexPos(r, q), DIRS[move - 1])
            

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
    # def turn(self, color: PlayerColor, action: Action):
        self.turns += 1
        if self.epsilon > self.decomposed:
            self.epsilon -= self.decay  
        """
        Update the agent with the last player's action.
        """
        match action:
            case SpawnAction(cell):
                if color == self._color:
                    state = self.horus.scry()
                    self.horus.spawn(cell, sign=1)
                    r_pos, q_pos = cell
                    a = (r_pos * DIM + q_pos) * DIM
                    self.toth.retain(state, a, self.reward, self.horus.scry(), self.end)
                    self.learn()
                else:
                    self.horus.spawn(cell, sign=-1)

            case SpreadAction(cell, direction):
                if color == self._color:
                    state = self.horus.scry()
                    self.horus.spread(cell, direction, sign=1)
                    r_pos, q_pos = cell
                    for i in range(0, len(DIRS)):
                        if DIRS[i] == direction:
                            a = (r_pos * DIM + q_pos) * DIM + i
                            break                    
                    self.toth.retain(state, a, self.reward, self.horus.scry(), self.end)
                    self.learn()
                else:
                    self.horus.spread(cell, direction, sign=-1)
    
    def upload(self):
        self.eveAluate.upload()
        print("upload successful")
    
    def printeresting_indeed(self):
        print("Batch update num: {}".format(self.updates))
        print("avg TD Error: {}\n avg Gradient: {}".format(self.eveAluate.batch_info()))
        
    def index_actions(self, action):
        move = int(action) % (len(DIRS) + 1) 
        posidex = (int(action) - move) / DIM
        q_pos = int(posidex % DIM)
        r_pos = int((posidex - q_pos) / DIM)
        return r_pos, q_pos, move
        # if move == 0:
        #     return SpawnAction((r_pos, q_pos))
        # return SpreadAction(HexPos(r_pos, q_pos), DIRS[move - 1])

        

    def learn(self):
        if self.toth.frames_stored < self.minibatch_size:
            return
        else:
            states, actions, rewards, new_states, endCheck = self.toth.ruminate(self.minibatch_size, encoder=self.horus)
        

        q_Ahead_offline = self.naiEve.batch_predict(new_states, self.minibatch_size)
        
        network_target = self.eveAluate.batch_predict(states, self.minibatch_size)
        
        q_Best_online_step = np.argmax(self.eveAluate.batch_predict(
                                        new_states, self.minibatch_size), axis=0)
        
        for state_i, isEnd in enumerate(endCheck):
            network_target[actions[state_i], state_i] = rewards[state_i] + \
                                                        self.gamma * \
                                                        (q_Ahead_offline[q_Best_online_step[state_i], state_i] * (1 - isEnd))
        
        self.eveAluate.batch_train(states, network_target, self.minibatch_size)

        self.updates+=1
        
        if self.updates % self.expulsion_period == 0:
            mse, grads = self.eveAluate.batch_info()
            self.mse.append(mse)
            self.grads.append(grads)
            self.naiEve.learn_from(self.eveAluate)
    
    def upload_errors(self):
        np.savez("errors_per_network_replace", mse=np.asarray(self.mse), grad=np.asarray(self.grads))

    
class Horus:
    def __init__(self):
        self.view = np.zeros((DIM, DIM), dtype=np.int8)

    def reset(self):
        self.view = np.zeros((DIM, DIM), dtype=np.int8)

    def scry(self):
        return self.view

    def spread(self, cell, dir, sign=0):
        r, q = cell
        pwr = abs(self.view[r, q].item())
        for i in range(1, pwr+1):
            pos = (HexPos(r,q) + dir * i)
            eval = self.view[pos.r][pos.q]
            if eval == MAX_POW : 
                eval = 0
            else:
                eval = abs(eval) * sign + sign
            self.view[pos.r][pos.q] = eval        
        self.view[r][q] = 0    
    
    def spawn(self, cell, sign=0):
        r, q = cell
        self.view[r][q] = sign
        return
    
    def sum_R(self):        
        return np.sum(np.greater(self.view, 0) * 1)

    def sum_B(self):
        return np.sum(np.less(self.view, 0) * 1)


    def encode(self, buffer=None, mode="seq"):
        if mode == "batch":
            view = buffer
            size = buffer.shape[0]

        else:
            view = np.reshape(self.view, (1, 7, 7))
            size = 1
        encoded = np.empty((size, 4, 13, 13), dtype=np.float32)


        for i in range(size):
            encoded[i, 0][:,:] = np.pad(np.greater(view[i], 0).astype(np.float32), 3, "wrap")
            encoded[i, 3][:,:] = np.pad(np.less(view[i], 0).astype(np.float32), 3, "wrap")


            encoded[i, 1][:,:] = np.zeros((13, 13), dtype=np.float32)
            row_self, col_self = np.nonzero(encoded[i][0])
            for j in range(row_self.size):
                pwr = view[i, row_self[j], col_self[j]]
                for dir in DIRS:
                    if pwr.item() >= int(pwr.item()) :
                        continue
                    for k in range(1, int(pwr.item()) + 1):
                        trace = (HexPos(row_self[j], col_self[j]) + dir * k)
                        encoded[i][1][trace.r][trace.q] = 1


            encoded[i, 2][:,:] = np.zeros((13, 13), dtype=np.float32)
            row_opp, col_opp = np.nonzero(encoded[i][3])
            for j in range(row_opp.size):
                pwr = view[i][row_opp[j]][col_opp[j]]
                for dir in DIRS:
                    if pwr.item() >= int(pwr.item()):
                        continue
                    for k in int(1, int(pwr).item() + 1):
                        trace = (HexPos(row_opp[j], col_opp[j]) + dir * k)
                        encoded[i][2][trace.r][trace.q] = 1

        if size == 1:
            return np.reshape(encoded, (4,13,13))
        return encoded

class Toth:
    def __init__(self, buffer_size:int, phi_size:int):
        self.buffer_size = buffer_size
        self.frames_stored = 0
        self.s = np.zeros((self.buffer_size, phi_size, phi_size), dtype=np.float32)
        self.a = np.zeros(self.buffer_size, dtype=np.int16)
        self.r = np.zeros(self.buffer_size, dtype=np.float16)
        self.s_ = np.zeros((self.buffer_size, phi_size, phi_size), dtype=np.float32)
        self.endCheck = np.zeros(self.buffer_size, dtype=np.bool_)
        return

    def retain(self, s, a, r, s_, endCheck):
        idx = self.frames_stored % self.buffer_size
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s_[idx] = s_
        self.endCheck[idx] = endCheck
        self.frames_stored += 1
        return
    
    def ruminate(self, depth, encoder=None):
        mem = np.random.choice(min(self.frames_stored, self.buffer_size), depth, replace=False)
        s = self.s[mem]
        a = self.a[mem]
        r = self.r[mem]
        s_ = self.s_[mem]
        endCheck = self.endCheck[mem]
        return encoder.encode(buffer=s, mode="batch"), a, r, encoder.encode(buffer=s_, mode="batch"), endCheck
