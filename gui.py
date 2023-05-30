import tkinter as tk
from tkinter import font
from tkinter import messagebox

from game import Game
from RL.model import RL_Model
from MCTS.model import MCTS_Model
from ANN.model import ANN_Model
from Minimax.model import Minimax_Model


class App(tk.Tk):

    def __init__(self):

        info = self.chose_type_of_game()

        super().__init__()
        self.title("Tic-Tac-Toe Game. By Javier Abollado.")
        self.f1 = TicTacToeBoard(self, info)
        self.f1.pack()

    def chose_type_of_game(self):
        
        print("\nChose your oponent:")
        print(" 1) Minimax")
        print(" 2) Artificial Neural Network")
        print(" 3) Reinforcement Learning")
        print(" 4) MCTS")
        option = input(" > ")

        mcts = False
        if option == "1":
            model  = Minimax_Model()
        elif option == "2":
            model  = ANN_Model("ANN/models/model.pkl")#"ANN/models/model_lr01_x20k_epochs100.pkl")
        elif option == "3":
            model  = RL_Model()
        else:
            model  = MCTS_Model()
            mcts = True

        print("\nStart player?")
        print(" 1) AI Player")
        print(" 2) Human")
        option = input(" > ")

        if option == "1":
            start_player = "AI"
            AI_chip = "x"
        else:
            start_player = "Human"
            AI_chip = "o"

        return model, start_player, AI_chip, mcts

class TicTacToeBoard(tk.Frame):

    def __init__(self, master, info):
        super().__init__(master=master)

        self.model, self.start_player, self.AI_chip, self.is_mcts = info

        # init variables and create frames
        self._cells = {}
        self._create_board_display()
        self._create_board_grid()

        # create a parallel board & load the AI model
        self.game = Game()

        if self.start_player == "AI":
            self.playAI()

    """
        Play our AI model (trained with Q learning)
         - 'playAI' makes a move (by the AI)
         - 'g' converts the turn : {"x","o"} -> {-1,1} in order to let the model read it
    """
    
    def playAI(self):
        a = self.model.move(self.game, only_action=True)
        self.move(a//3, a%3)

    """
        Move functions
         - 'command' is a function that returns another function 'move' to the button (row, col)
         - 'move' checks & print the move, also check if we have won or not  
    """

    def command(self, row, col):
        frc = lambda : self.move(row, col)
        return frc
    
    def move(self, row, col):
        button = self._cells[(row,col)]
        if button["text"] == "": # and not self.win:
            button.config(text=self.game.player)
            action = 3*row+col
            # update the mcts in case
            if not self.game.player == self.AI_chip and self.is_mcts and self.game.empty_spaces > 1:
                self.model.move_direct(action)
            self.game.insert(action)
            # check winner 
            finished = self.game.finished(type=True)
            if finished != -1:
                if finished == 0:
                    self.draw_row(row)
                elif finished == 1:
                    self.draw_col(col)
                elif finished == 2:
                    self.draw_diag1()
                else:
                    self.draw_diag2()
                self.winner()
            elif self.game.full():
                self.winner(draw=True)
            else:
                # change turn after move
                self.display.config(text=f"Turn : {self.game.player}")
                if self.game.player == self.AI_chip and self.game.empty_spaces > 0:
                    self.playAI()

    """ 
        1) Create a header (self.display)
        2) Create a 3x3 board 
    """

    def _create_board_display(self):
        display_frame = tk.Frame(master=self)
        display_frame.pack(fill=tk.X)
        self.display = tk.Label(
            master=display_frame,
            text="Ready?",
            font=font.Font(size=28, weight="bold"),
        )
        self.display.pack()

    def _create_board_grid(self):
        grid_frame = tk.Frame(master=self)
        grid_frame.pack()
        for row in range(3):
            # self.rowconfigure(row, weight=1, minsize=50)
            # self.columnconfigure(row, weight=1, minsize=50)
            for col in range(3):
                button = tk.Button(
                    master=grid_frame,
                    command=self.command(row,col),
                    text="",
                    font=font.Font(size=36, weight="bold"),
                    fg="black",
                    width=4,
                    height=2,
                    relief="ridge",
                    borderwidth=4,
                    highlightbackground="lightblue",
                )
                self._cells[(row, col)] = button
                button.grid(
                    row=row,
                    column=col,
                    padx=5,
                    pady=5,
                    sticky="nsew"
                )
                
    def reset_game(self):
        self.display.config(text="Ready?")
        for i in range(3):
            for j in range(3):
                self._cells[(i,j)].config(text="", fg="black")
        self.game.reset()
        if self.is_mcts:
            self.model.reset()
        if self.start_player == "AI":
            self.playAI()

    """ 
        1) Check if someone wins : ROWS / COLUMNS / DIAGONALS 
        2) Draw with green the line which has won the game
    """
            
    def winner(self, draw=False):
        self.game.change_player()
        if not draw:
            message = f"Player '{self.game.player}' wins!\nReady for another round?"
        else:
            message = f"Draw!\nReady for another round?"
        x = messagebox.askyesno(message=message, title="End Game!")
        self.game.change_player()
        if x:
            self.reset_game()
        else:
            exit(0)

    def draw_row(self, row):
        for j in range(3):
            self._cells[(row,j)].config(fg="green")

    def draw_col(self, col):
        for i in range(3):
            self._cells[(i,col)].config(fg="green")

    def draw_diag1(self):
        for k in range(3):
            self._cells[(k,k)].config(fg="green")

    def draw_diag2(self):
        for k in range(3):
            self._cells[(k,2-k)].config(fg="green")
