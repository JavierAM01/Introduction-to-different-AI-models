import copy
from pickle_files import save

"""
    We create all the posibles boards that we can have in a TicTacToe game (6064 to be more specific).
    We save the results in the set 'all'
    That number comes from:

        sum_{n=0}^{9} [ comb(9,n) * comb(n, n//2) ] = 6064

    Represents the sum of all the posible combinations of 'n' chips on the board, comb(9,n), times the number of
    posible ways that can be permutated between the 'x' and the 'o' player in that positions, comb(n, n//2).
"""
def create_all_posible_boards():
    global all
    all = set([(0,0,0,0,0,0,0,0,0)])
    _create_all_posible_boards(l=[0,0,0,0,0,0,0,0,0], n=0)
    return all

def _create_all_posible_boards(l, n):
    for i in range(n,9):
        if l[i] == 0:
            for turn in [-1,1]:
                l[i] = turn
                if sum([1 for e in l if e == 1]) - sum([1 for e in l if e == -1]) in [0,1]: 
                    all.add(copy.deepcopy(tuple(l)))
                _create_all_posible_boards(l, n+1)
                l[i] = 0


if __name__ == "__main__":
    all = create_all_posible_boards()
    save("data/boards.pkl", all)