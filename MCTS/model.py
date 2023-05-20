import numpy as np

class MCTS_Model:

    def __init__(self):
        pass



"""
    Class to make a perfect move doing a full tree search.
    With the 'move' function we can return the next action to do given a certain state. 
"""
class Minimax:

    # find all the posibilities and complete the self.Ns list with winners and looses
    def minimax(self, game, a0=None):
        # save actual info
        b0 = a0
        player = game.player
        puntuation = [(-np.inf,0) for _ in range(9)]
        for a in range(9):
            if not game.valid(a):
                continue
            # move
            game.insert(a)
            # score for this action
            # The score is a tuple:
            #  - first component : actual score € [0,10]
            #  - second component: preference in case of a draw finding the best score -> 2 (win), 1 (draw), 0 (lose)
            score = None
            # check winner
            if a0 == None:
                a0 = a
            # someone has won
            if game.finished():
                if game.player == "o":
                    score = (game.empty_spaces + 1, 2)
                else:
                    score = (-10, 0)
            # draw
            elif game.full():
                score = (game.empty_spaces + 1, 1)
            # continue
            else:
                _, score = self.minimax(game, a0)
            # reset game before the move
            game.erase(a)
            game.player = player
            a0 = b0
            puntuation[a] = score
        # select action and giving score
        if (-10,0) in puntuation:
            best_score = (-1,0) # -10 indicates that we lose in that move, while -1 indicates that we dont want to play that move because then we lose
            best_action = -1
        else:
            best_score = max(puntuation)
            best_action = puntuation.index(best_score)
        return best_action, best_score
        
    def get_best_action(self, game): 
        a, _ = self.minimax(game)
        a = a if a != -1 else 0
        return a




