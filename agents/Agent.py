import numpy as np
import math
from absl import logging


class FIFO(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        return 0

class LRU(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        return np.argmin(obs['req_time'])

class LFU(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        return np.argmin(obs['freq'])

class SIZE(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        size_max = np.argwhere(obs['size'] == np.amax(obs['size'])).flatten().tolist()
        action = obs['req_time'].index(min([obs['req_time'][i] for i in size_max]))
        return action

class belady(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        return np.argmax(obs['next_req'])

class belady_size(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        a = obs['next_req']
        b =  obs['size']
        mul_ = [a[i] * b[i] for i in range(len(a))]
        return np.argmax(mul_)

class Random(object):
    def __init__(self):
        pass
    def choose_action(self, obs):
        return np.random.randint(len(obs['item']))

class m_metric(object):
    def __init__(self):
        self.pf = 1
        self.pr = -0.5
        self.ps = -1
    def choose_action(self, obs):
        min_index = 0
        min_value = 1e9
        for i in range(len(obs['freq'])):
            # print(obs['freq'][i], obs['rect'][i], obs['size'][i])
            # t = math.pow(obs['freq'][i], self.pf) * math.pow(obs['rect'][i], self.pr) * math.pow(obs['size'][i], self.ps)
            t = math.pow(obs['freq'][i], self.pf) * math.pow(obs['rect'][i], self.pr)
            if t < min_value:
                min_value = t
                min_index = i
        return min_index