from pickle_files import load, save


"""
    Map the code of every single posible board with a number
"""
def create_codes():
    all = load("data/boards.pkl")
    # keys = list(map(get_code, all))
    codes = dict(zip(all, range(6064)))
    save("RL/models/codes.pkl", codes)








###################################################################################################################

# """
#     Example of how it works: 
#     l = [1,0,0,-1,-1,1,0,1,-1]
#     => separate
#         [0,0,0, 1, 1,0,0,0, 1]  --> ones where there is a '-1'
#         [1,0,0, 0, 0,1,0,1, 0]  --> ones where there is a '+1'
#     => to binary
#     s1 = sum([2**k for k in range(9) if l[k] == -1])
#     s2 = sum([2**k for k in range(9) if l[k] == +1])
#     => results
#         (s1, s2)
# """
# def get_code(tupla):
#     res = [0,0]
#     for k in range(9):
#         if tupla[k] != 0:
#             i = (tupla[k]+1)//2 # {-1,1} -> {0,1}
#             res[i] += 2**k
#     return tuple(res)

# """
#     If we give a board instead of the tuple, first we convert it to tuple and then we call the other function
# """
# def get_code_from_board(board):
#     tupla = tuple(board.reshape((9,)))
#     return get_code(tupla)


#################################################################################################################

# """
#     In this type of initialization we add to the Q-table some information.
#     First we initializate the Q-table with random values between 0 and 0.1. Then
#     every move that leads us to win or either lose if we dont do it, we apply 1 to that action
#     and 0's to the others, so now we know only how to do the final step. Then after trainning the model 
#     we spect to learn somehow to move from the beginning in an optimal way.
# """
# def init_Qtable_type2(seed=0):
#     np.random.seed(seed)
#     n = 6064    # len(codes)
#     # q table
#     q_table = np.random.random((n,9)) / 10 # inicialize with random values between (0, 0.1)
#     for b in all:   # initialize with 1 those actions which lead us to win or avoid lossing
#         player = 1 if sum(b) == 0 else -1
#         state = list(b)
#         pos = check_winner(state, player)
#         # winning
#         if pos != (-1,-1):
#             action = 3*pos[0] + pos[1]
#             code = get_code(state)
#             i = codes[code]
#             q_table[i] = np.array([(1 if a == action else 0) for a in range(9)])
#         # avoid lossing
#         else:
#             player2 = 1 if player == -1 else -1
#             pos = check_winner(state, player2)
#             if pos != (-1,-1):
#                 action = 3*pos[0] + pos[1]
#                 code = get_code(state)
#                 i = codes[code]
#                 q_table[i] = np.array([(1 if a == action else 0) for a in range(9)])
#     return codes, q_table

# """
#     Checks if with that state any of the players have won the game
# """
# def check_winner(state, player):
#     # rows
#     for i in range(3):
#         if sum(state[3*i:3*(i+1)]) == 2 * player:
#             for j in range(3):
#                 if state[3*i+j] == 0:
#                     return (i,j)
#     # columns
#     for j in range(3):
#         if sum(state[j::3]) == 2 * player:
#             for i in range(3):
#                 if state[3*i+j] == 0:
#                     return (i,j)
#     # diagonal1
#     if (state[0] + state[4] + state[8]) == 2 * player:
#         for i,j in [(0,0), (1,1), (2,2)]:
#             if state[3*i+j] == 0:
#                 return (i,j)
#     # diagonal2
#     elif (state[2] + state[4] + state[6]) == 2 * player:
#         for i,j in [(0,2), (1,1), (2,0)]:
#             if state[3*i+j] == 0:
#                 return (i,j)
    
#     return (-1,-1)