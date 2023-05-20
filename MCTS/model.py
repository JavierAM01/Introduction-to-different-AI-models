import numpy as np
from MCTS.mcts import MCTS

from game import Game

class MCTS_Model:

    def __init__(self):
        game = Game()
        self.mcts = MCTS(game)

        # make a serious number of simulations to train the model, once its trained we can use it
        for _ in range(10 * 362880): # 362880 = 9!, which represents the  Nº of posibles leafs in the tree -> in order to check most of them
            self.mcts.simulation()

        self.current_node = self.mcts

    def move(self, game, only_action=False):
        a = self.current_node.move(game)
        game.insert(a)
        finished = game.finished()
        if not finished:
            self.current_node = self.current_node.edges[a].nextState
        if only_action:
            game.erase(a)
            return a

    def reset(self):
        self.current_node = self.mcts

    """
        If we create a way to save the model (insted of train it every time is created) 
        then we create functions to save it and load it.
    """
    def load(self):
        pass

    def save(self):
        pass



