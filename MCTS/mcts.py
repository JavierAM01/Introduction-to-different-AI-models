import math
import copy
import numpy as np
import torch as T

from game import Game


NUM_ACTIONS = 9
NUM_SIMULATIONS = 200
NUM_SIMULATIONS_3 = 200
C_PUCT = 1.0
eps = 0.25   # 0 to play match games, 0.25 to train
PROF = 9     # 0 to play match games, 9 to train
eps_erase = 0.25
PROF_erase = 9

DIR_ALPHA = 1.5

from MCTS.edge import Edge     
        

# original to train the net, make a random rotation each step 
class MCTS:
        
    def __init__(self, game):
        available = game.available_moves()
        p = 1.0 / len(available)
        #dirich = np.random.dirichlet([DIR_ALPHA for _ in range(NUM_ACTIONS)], 1)[0]
        probs = [p if i in available else 0 for i in range(9)]
        self.edges = [Edge(p) for p in probs]
        self.board = copy.deepcopy(game.board)
        self.player = game.player
        self.empty_spaces = game.empty_spaces

    """Create a new Game equivalent to this node"""
    def create_game(self):
        new_game = Game()
        new_game.board = copy.deepcopy(self.board)
        new_game.player = self.player
        new_game.empty_spaces = self.empty_spaces
        return new_game

    """
        Make one simulation: we go throught all the nodes making decisions, until we reached a leaf node. 
        Then we update de edges knowing the new information.
    """
    def simulation(self):
        
        current_node = self
        current_game = self.create_game()

        actions = []        
        
        while True:

            # Find next action: a = argmax{Q(a) + U(a)} 
            Q = map(lambda edge: edge.q, current_node.edges)
            sqrt_sum = math.sqrt(sum(map(lambda edge: edge.n, current_node.edges)))
            U = [(C_PUCT * edge.p * sqrt_sum / (1 + edge.n)) for edge in current_node.edges]
            
            # Avoid unavailable actions 
            possible_actions = current_game.available_moves()
            confidence = [(q+u if i in possible_actions else -np.inf) for i,(q,u) in enumerate(zip(Q,U))]
            action_taken = np.argmax(confidence)
            actions.append(action_taken)
            
            # Update the current game
            current_game.insert(action_taken)
            finished = current_game.finished()
            
            # We haven't reached a leaf node yet
            if current_node.edges[action_taken].nextState != None:
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState 
                # # This action its not necesary because the board updates it self:
                # current_game.board = copy.deepcopy(current_node.board)
            
            # We have reached a leaf node, so we have to expand it
            else:
                # Winner
                if finished:
                    if len(actions) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                # Draw
                elif current_game.full():
                    total_value = 0
                # Nothing
                else:
                    # # Evaluate: model(state) -> value & policy 
                    # # ROTATIONS -> maybe we can try some rotations to the model. 
                    # r = Rotation()
                    # new_board = r.rotate_random(current_game.board)
                    # current_game.board = new_board
                    total_value = 0
                    # if len(actions) % 2 == 1:
                    #     total_value = -total_value
                    # Create next Node
                    current_node.edges[action_taken].nextState = MCTS(current_game)
                    
                break
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for action in actions:
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            total_value = -total_value
            
    """
        Make a move: first we make some simulations (even though we have already trained the model) and then we select the best action.
    """
    def makeMove(self, game, n_games=15):

        # Run simulations to populate our MCTS
        for i in range(NUM_SIMULATIONS):
            if i % 100 == 0:
                print(i)
            self.simulation()

        # # Uncomment to see the number of simulations made per action
        # print(list(map(lambda edge: edge.n, self.edges)))
        
        # Choose action depending on the simulations
        if n_games < 15:
            TAU = 1    # Choose the value of tau depending on the current depth
            sum_tot = sum([edge.n**(1/TAU) for edge in self.edges])
            probs = [(edge.n**(1/TAU)) / sum_tot for edge in self.edges]
            while True:
                action = np.random.choice(a=list(range(NUM_ACTIONS)), p=probs)
                if action in game.available_moves():
                    break
        else:
            action = np.argmax(list(map(lambda edge: edge.n, self.edges)))

        return action

