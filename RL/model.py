from game import Game
import copy
import numpy as np
import random
from pickle_files import load, save
import os
import matplotlib.pyplot as plt


main_path = "RL/models/"

# # => Copy this code to train a model (from parent)
# from RL.model import Q_Trainner
# name = "q_100k_e1.pkl"  # -> 100k = number of epochs, e1 = epision 0.1
# trainner = Q_Trainner(epsilon=0.1)
# hist = trainner.train(save_path=name, N=100000)

class RL_Model: # Q-Learning

    def __init__(self, path_qtable="q_100k_e1.pkl"):
        self.codes = load(os.path.join(main_path, "codes.pkl"))
        self.table = load(os.path.join(main_path, path_qtable))

    def valid(self, board, action):
        return (board[action // 3, action % 3] == 0)

    def get_code(self, board):
        return tuple(board.reshape((9,)))

    def chose_best_action(self, board):
        code = self.get_code(board) #self.board2code(board)
        q_values = self.table[self.codes[code]]
        posibilities = np.array([i for i in range(9) if self.valid(board, i)])
        weights = q_values[posibilities]
        # we want to return de index of the best q-value
        i = np.argmax(weights)
        action = posibilities[i]
        return action

    # Make a move, guided by the best action finded on the Q-table
    def move(self, game, only_action=False):
        a = self.chose_best_action(game.board)
        if only_action:
            return a
        game.insert(a)


#######################################################################################################################
#######################################################################################################################
#####                                                TRAINNER                                                     #####
#######################################################################################################################
#######################################################################################################################

class Q_Trainner:

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.codes = load(os.path.join(main_path, "codes.pkl"))
        self.table = np.zeros((6064,9), dtype=np.float32)
        self.t = 0

    def reward(self, state, action):
        return 0 # we are going to initialize the values directly in the Q table / so we dont need any reward during training

    def valid(self, board, action):
        return (board[action // 3, action % 3] == 0)

    def next_game(self, game, action):
        next_game = copy.deepcopy(game)
        next_game.insert(action)
        return next_game

    def get_code(self, board):
        return tuple(board.reshape((9,)))

    def chose_action(self, board):
        code = self.get_code(board)
        q_values = self.table[self.codes[code]]
        posibilities = np.array([i for i in range(9) if self.valid(board, i)])
        weights = q_values[posibilities]
        if np.random.random() < (1 - self.epsilon) - self.t * (1 - 2*self.epsilon):
            action = np.random.choice(posibilities)
        else:
            i = np.argmax(weights)
            action = posibilities[i]
        return action

    def chose_action_type2(self, board):
        code = self.get_code(board)
        q_values = self.table[self.codes[code]]
        posibilities = np.array([i for i in range(9) if self.valid(board, i)])
        weights = q_values[posibilities]
        action = random.choices(posibilities, weights=weights, k=1)[0]
        return action

    def chose_best_action(self, board, q=False):
        code = self.get_code(board)
        q_values = self.table[self.codes[code]]
        posibilities = np.array([i for i in range(9) if self.valid(board, i)])
        weights = q_values[posibilities]
        # if we want to return the best q-values
        if q:
            return max(weights)
        # else -> we want to return de index of the best q value
        i = np.argmax(weights)
        action = posibilities[i]
        return action

    # we didn't separete the cases in which we lose or win after 2 moves...
    def updateQtable(self, game, action):
        board = game.board
        # actual values
        code = self.get_code(board)
        reward = self.reward(board, action)
        q0 = self.table[self.codes[code]][action]
        # print(action)
        # print(board)
        # check if we won
        next_game = self.next_game(game, action)
        if next_game.finished():
            q_next = 1
            # print(q_next)
            # print("Finished!")
            # print(next_game.board)
        elif next_game.full():
            q_next = 0
            # print(q_next)
            # print("Draw!")
            # print(next_game.board)
        else:
            # next move -> max Q for the enemy
            a = self.chose_best_action(next_game.board)
            # check if we lose
            next_game = self.next_game(next_game, a)
            if next_game.finished():
                q_next = -1
                # print(q_next)
            elif next_game.full():
                q_next = 0
                # print(q_next)
            else:
                # max Q for me in my next move 
                q_next = self.chose_best_action(next_game.board, q=True)
        # final Q max (2 steps after)
        q_new = (1 - self.alpha) * q0 + self.alpha * (reward + q_next)
        self.table[self.codes[code]][action] = q_new

    # play one simulation of the game from the beginning
    def play_vs_itself(self):
        game = Game()
        isFull = False
        isFinished = False
        while not isFull and not isFinished:
            a = self.chose_action(game.board)
            self.updateQtable(game, a)
            game.insert(a)
            isFull = game.full()
            isFinished = game.finished()
        if not isFinished:
            return 0
        winner = game.get_chip()
        return winner

    # play one simulation of the game from the beginning
    def play_vs_model(self, model):
        game = Game()
        AI_turn = False
        isFull = False
        isFinished = False
        while not isFull and not isFinished:
            if not AI_turn:
                a = self.chose_action(game.board)
                self.updateQtable(game, a)
                game.insert(a)
            else:
                model.move(game)
            AI_turn = not AI_turn
            isFull = game.full()
            isFinished = game.finished()
        if not isFinished:
            return 0
        winner = game.get_chip()
        return winner

    """
        Play N simulations and save the results -> if save_path != None => save the model
        How we select randomly the actions. Supose that epsilon is 0.1, then:
        
        First we start with a '1 - epsilon' (1 - 0.1 = 0.9) chance of select a random action. At the end
        of the trainning we select randomly with a probability of 'epsilon' (0.1). This change goes
        changes linearly (with the help of 'self.t € [0,1]' that goes from 0 and every step goes closer to 1)
    """
    def train(self, N=100, save_path=None, model=None):
        wins = 0
        draws = 0
        loss = 0
        historial = []
        for n in range(N):
            self.t = min(n / N, 1)
            if n % 1000 == 0:
                historial.append((wins, draws, loss))
                print(f"[{n}] wins {wins} | draws {draws} | losses {loss}")
            winner = self.play_vs_itself()
            if winner == 0:
                draws += 1
            elif winner == 1:
                wins += 1
            else:
                loss += 1
        self.t = 0
        hist = np.array(historial)
        if save_path != None:
            path = os.path.join(main_path, save_path)
            save(path, self.table)
            self.plot_hist(hist, path)
        return hist

    def plot_hist(self, hist, save_path):
        N = len(hist)
        x = range(N)
        plt.title("Trainning results")
        plt.plot(x, hist[:,0], "g-", label="wins")
        plt.plot(x, hist[:,1], "k-", label="draws")
        plt.plot(x, hist[:,2], "r-", label="losses")
        plt.xlabel("epoch (x1000)")
        plt.ylabel("n")
        plt.legend()
        plt.savefig(save_path[:-4] + ".png")
        plt.show()


