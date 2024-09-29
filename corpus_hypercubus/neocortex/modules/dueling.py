from . import xp, BaseLayer, Dense


class Dueling(BaseLayer):
    def __init__(self, phi_size, q_size, name="dueling"):
        super().__init__(name=name)
        
        self.value = Dense(phi_size, 1, name="value")
        self.advantage = Dense(phi_size, q_size, name="advantage")

    def forward(self, phi, mode="quality"):
        self.phi = phi
        if mode == "quality":
            a = self.advantage.forward(self.phi)
            return (self.value.forward(self.phi) + (a - xp.mean(a)))
        else:
            return self.advantage.forward(self.phi)

    def backpropagate(self, dE_dV, optimiser:object):

        dE_dPhi_A = self.advantage.backpropagate(dE_dV, optimiser)
        dE_dPhi_V = self.value.backpropagate(xp.reshape(xp.mean(dE_dV), (1,1)), optimiser)
        return ((dE_dPhi_A + dE_dPhi_V) / xp.sqrt(2))

    def link(self, dE_dV=0, insight=None, mode="transfer"):
        if mode == "transfer":
            temp = {}
            temp.update(self.value.link(mode=mode))
            temp.update(self.advantage.link(mode=mode))
            return temp
        
        elif mode == "receive":
            self.value.link(insight, mode=mode)
            self.advantage.link(insight, mode=mode)
            return 