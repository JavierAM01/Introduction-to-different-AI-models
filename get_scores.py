import random
import numpy as np
import matplotlib.pyplot as plt

from RL.model import RL_Model
from MCTS.model import MCTS_Model
from ANN.model import ANN_Model
from Minimax.model import Minimax_Model

from game import Game

from pickle_files import load

# load diferent kind of rules
rules1 = load("Minimax/models/rules.pkl")

rules2 = load("Minimax/models/rules.pkl")
rules2[tuple([0]*9)] = 4 # start in the center

# random move
def move_random(board):
    posibilities = [i for i in range(9) if board[i] == 0]
    return random.choice(posibilities)

# minimax move
def move_well(board):
    return rules1[board]



def play(model, mode="random", start_AI=False, is_mcts=False):

    game = Game()

    move = move_random if mode == "random" else move_well

    win  = 0
    lose = 0
    draw = 0

    i = 0 if start_AI else 1
    while True:

        if i % 2 == 0:
            model.move(game)
        else:
            t = tuple(game.board.reshape(-1))
            a = move(t)
            game.insert(a)
            if is_mcts and game.empty_spaces > 1:
                model.move_direct(a)
        if game.finished():
            if i % 2 == 0:
                win = 1
            else:
                lose = 1
            break
        if game.full():
            draw = 1
            break

        i += 1
    
    return win, lose, draw




def main():

    rl_model = RL_Model()
    mcts_model = MCTS_Model()
    ann_model = ANN_Model("ANN/models/model.pkl")

    N = 10
    x = range(N+1)

    fig_rl = plt.figure()
    fig_rl.suptitle('RL Model')
    fig_ann = plt.figure()
    fig_ann.suptitle('ANN Model')
    fig_mcts = plt.figure()
    fig_mcts.suptitle('MCTS Model')

    modes = ["random", "random", "other", "other"]
    start_AI = [True, False, True, False]
    titles = ["AI vs Random - AI starts", "AI vs Random - Random starts", "AI vs Player - AI starts", "AI vs Player - Player starts"]

    def make_list(model, i, is_mcts=False):
        wins   = [0]
        losses = [0]
        draws  = [0]
        for _ in range(N):
            win, lose, draw = play(model, mode=modes[i], start_AI=start_AI[i], is_mcts=is_mcts)
            wins.append(wins[-1] + win)
            losses.append(losses[-1] + lose)
            draws.append(draws[-1] + draw)
            if is_mcts:
                model.reset()
        return wins, losses, draws
    
    for i in range(4):

        # RL
        wins, losses, draws = make_list(rl_model, i)
        ax = fig_rl.add_subplot(2, 2, i+1)
        ax.plot(x, wins, "g-", label="wins")
        ax.plot(x, losses, "r-", label="losses")
        ax.plot(x, draws, "k-", label="draws")
        ax.set_title(titles[i])
        ax.legend()

        # ANN
        wins, losses, draws = make_list(ann_model, i)
        ax = fig_ann.add_subplot(2, 2, i+1)
        ax.plot(x, wins, "g-", label="wins")
        ax.plot(x, losses, "r-", label="losses")
        ax.plot(x, draws, "k-", label="draws")
        ax.set_title(titles[i])
        ax.legend()

        # MCTS
        wins, losses, draws = make_list(mcts_model, i, True)
        ax = fig_mcts.add_subplot(2, 2, i+1)
        ax.plot(x, wins, "g-", label="wins")
        ax.plot(x, losses, "r-", label="losses")
        ax.plot(x, draws, "k-", label="draws")
        ax.set_title(titles[i])
        ax.legend()

    plt.show()



if __name__ == "__main__":
    main()


