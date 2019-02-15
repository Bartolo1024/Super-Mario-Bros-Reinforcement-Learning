import math

class Epsilon(object):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=20000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step = 0

    def __call__(self):
        eps_th = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.step / self.eps_decay)
        self.step += 1
        return eps_th