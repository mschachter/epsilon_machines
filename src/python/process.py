import numpy as np


class BinaryHMM(object):

    def __init__(self, T0, T1, num_states, initial_state_dist):
        self.M = num_states
        self.T0 = T0
        self.T1 = T1
        self.T = T0 + T1
        self.initial_state_dist = np.array(initial_state_dist)
        #compute stationary distribution
        Tinf = np.linalg.matrix_power(self.T, int(500))
        self.stationary_distribution = Tinf[0, :]

    def simulate(self, L=10000):
        #generate an initial state
        rnum = np.random.rand()
        s0 = (self.initial_state_dist.cumsum() < rnum).argmin()
        output = list()
        sk = s0
        for k in range(L):
            csum = np.cumsum(self.T[sk, :])
            rnum = np.random.rand()
            s = (csum < rnum).argmin()
            if self.T0[sk, s] > 0:
                output.append(0)
            elif self.T1[sk, s] > 0:
                output.append(1)
            sk = s
        return np.array(output)


class GoldenMean(BinaryHMM):

    def __init__(self):
        T0 = np.array([[0.0, 0.5], [0.0, 0.0]])
        T1 = np.array([[0.5, 0.0], [1.0, 0.0]])
        BinaryHMM.__init__(self, T0, T1, 2, [0.5, 0.5])


class EvenProcess(BinaryHMM):

    def __init__(self):
        T0 = np.array([[0.5, 0.0], [0.0, 0.0]])
        T1 = np.array([[0.0, 0.5], [1.0, 0.0]])
        BinaryHMM.__init__(self, T0, T1, 2, [0.5, 0.5])


class BiasedCoin(object):

    def __init__(self, p=0.5):
        """ p = probability of heads """
        self.p = p

    def simulate(self, L=10000):
        samps = np.zeros([L])
        for k in range(L):
            r = np.random.rand()
            if r < self.p:
                samps[k] = 1
        return samps
