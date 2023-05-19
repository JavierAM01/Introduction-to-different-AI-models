import numpy as np
from game import Game
from pickle_files import load, save


"""
    Function to create all the rules for every posible board combination in the TicTacToe Game.
    Arguments =>
    The path to save the rules (a dictionary)
    Return => 
    Dictionary:
        - key: tuple [dim = (9,)] which represents the board
        - value: action to make 
"""
def create_rules(save_path="Perfect/models/rules.pkl"):
    rules = dict()
    all = load("data/boards.pkl")
    for _tuple in all:
        game = tuple2game(_tuple)
        a = get_best_action(game)
        rules[_tuple] = a
    save(save_path, rules)


"""
    Function to transform a tuple (which represents a TicTacToe board) to an actual Game object.
"""
def tuple2game(_tuple):
    empty_spaces = sum([1 for e in _tuple if e == 0])
    board = np.array(_tuple, dtype=np.int8)
    board = board.reshape((3,3))
    game = Game()
    game.board = board
    game.player = "x" if board.sum() == 0 else "o"
    game.empty_spaces = empty_spaces
    return game



"""
    Funtion to make a perfect move doing a full tree search. 
"""
def get_best_action(game):
    if game.board[1][1] == 0:
        a = 4
    else:
        _, _, a = minimax(game)
    return a

def minimax(game):  # return (X, nº of wins, nº of defeats) where X = 0 -> draw / 1 -> win / -1 -> lose
    
    # case 0
    if game.finished():
        return (1, game.empty_spaces + 1, None)
    if game.full():
        return (0, 0, None)
    
    x1 = [-np.inf for _ in range(9)]  # results
    x2 = [0 for _ in range(9)]        # wins

    n_loses = 0

    for a in range(9):
        if not game.valid(a):
            continue

        game.insert(a)
        
        res, n1, _action = minimax(game)

        x1[a] = res
        x2[a] = n1

        if res == -1:
            n_loses += 1
        
        game.erase(a)

    # get best action -> in case of a draw we select the action who has more wins.
    i = np.argmax(x1)
    if x1[i] < 1:
        index = [j for j in range(9) if x1[j] == x1[i]]
        j = np.argmax([x2[j] for j in index])
        i_max = index[j]
    else:
        i_max = i
    
    # Si el actual gana el siguiente pierde (y viceversa)
    # El nº de ganancias del actual, son las pérdidas del siguiente (y viceversa)

    return x1[i_max] * (-1), n_loses, i_max

