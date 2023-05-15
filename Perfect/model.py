import numpy as np
from pickle_files import load


class Perfect_Model:

    def __init__(self):
        self.rules = load("Perfect/models/rules.pkl")

    def move(self, game, only_action=False):
        _tuple = tuple(game.board.reshape((9,)))
        a = self.rules[_tuple]
        if only_action:
            return a
        game.insert(a)