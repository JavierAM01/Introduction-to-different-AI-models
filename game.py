import numpy as np

class Game_Info:

    def __init__(self):
        self.board = np.zeros((3,3), dtype=np.int8)
        self.player = "x"
        self.empty_spaces = 9

class Game(Game_Info):

    def reset(self):
        self.__init__()

    def set_player(self, player):
        self.player = player

    def change_player(self):
        self.player = "x" if self.player == "o" else "o"

    def get_chip(self):
        if self.player == "x":
            return 1
        return -1
    
    def available_moves(self):
        return [a for a in range(9) if self.valid(a)] 

    def valid(self, action):
        return self.board[action//3,action%3] == 0

    def full(self):
        return self.empty_spaces == 0

    # 1) type = False: return if the game is finished or not
    # 2) type = True: return -1 is the game is not finished and {0,1,2,3} if it is (depending on the type)
    def finished(self, type=False):
        # check rows
        for i in range(3):
            if abs(self.board[i].sum()) == 3:
                return (True if not type else 0)
        # check columns
        for j in range(3):
            if abs(self.board[:,j].sum()) == 3:
                return (True if not type else 1)
        # check diagonals
        if abs(sum([self.board[i,i] for i in range(3)])) == 3:
            return (True if not type else 2)
        if abs(sum([self.board[2-i,i] for i in range(3)])) == 3:
            return (True if not type else 3)
        return (False if not type else -1)

    def insert(self, action):
        chip = self.get_chip()
        self.board[action // 3, action % 3] = chip
        self.empty_spaces -= 1
        self.change_player()

    def erase(self, action):
        self.board[action // 3, action % 3] = 0
        self.empty_spaces += 1
        self.change_player()