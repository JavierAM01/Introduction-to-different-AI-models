import tkinter as tk
from tkinter import font
from tkinter import messagebox

from game import Game
from RL.model import RL_Model
from MCTS.model import MCTS_Model
from ANN.model import ANN_Model
from Perfect.model import Perfect_Model


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Tic-Tac-Toe Game")
        self.f1 = TicTacToeBoard(self)
        self.f1.pack()
        # self.f2 = TicTacToeBoard(self)
        # self.f1.grid(row=0, column=0, padx=10)
        # self.f2.grid(row=0, column=1, padx=10)

class TicTacToeBoard(tk.Frame):

    def __init__(self, master):
        super().__init__(master=master)

        # init variables and create frames
        self._cells = {}
        self._create_board_display()
        self._create_board_grid()

        # create a parallel board & load the AI model
        self.game = Game()
        print("Chose a model:")
        print(" 1) RL Model")
        print(" 2) ANN Model")
        print(" 3) Perfect Model")
        print(" 4) MCTS Model (Not developed yet)")
        option = input(" > ")
        print(option)
        if option == "1":
            self.model  = RL_Model()
        elif option == "2":
            self.model  = ANN_Model("ANN/models/model_lr01_x20k_epochs100.pkl")
        elif option == "3":
            self.model  = Perfect_Model()
        else:
            self.model  = MCTS_Model()
        # self.model2 = Perfect_Model()
        self.AI_chip = "o"

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
            self.game.insert(3*row+col)
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
                if self.game.player == self.AI_chip:
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

    """ 
        1) Check if someone wins : ROWS / COLUMNS / DIAGONALS 
        2) Draw with green the line which has won the game
    """
            
    def winner(self, draw=False):
        if not draw:
            message = f"Player '{self.game.player}' has won the game!\n¿You want to restart the game?"
        else:
            message = f"Draw! Both players win!\n¿You want to restart the game?"
        x = messagebox.askyesno(message=message, title="Advertisement")
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


def main():
    """Create the game's board and run its main loop."""
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()