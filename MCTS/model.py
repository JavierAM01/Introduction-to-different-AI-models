import numpy as np
from MCTS.mcts import MCTS

from game import Game

class MCTS_Model:

    def __init__(self):
        game = Game()
        self.mcts = MCTS(game)
        self.current_node = self.mcts

        # make a serious number of simulations to train the model, once its trained we can use it
        print("Trainning model... Please wait ;)")
        for _ in range(10000): # 362880 = 9!, which represents the  Nº of posibles leafs in the tree -> in order to check most of them
            self.mcts.simulation(trainning=True)

    def move(self, game, only_action=False):
        a = self.current_node.move(game)
        game.insert(a)
        finished = game.finished()
        if not finished and not game.full():
            self.move_direct(a)
        if only_action:
            game.erase(a)
            return a
        
    def move_direct(self, a):
        if self.current_node.edges[a].nextState == None:
            game = self.current_node.create_game()
            game.insert(a)
            self.current_node.edges[a].nextState = MCTS(game)
            print("Unexplore table!")
        self.current_node = self.current_node.edges[a].nextState
        # to prevent unexplore tables we retrainned the model each movement 
        for _ in range(1000):
            self.current_node.simulation(trainning=True)

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



