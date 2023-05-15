import math
import copy
import numpy as np
import torch as T


NUM_ACTIONS = 9
NUM_SIMULATIONS = 200
NUM_SIMULATIONS_3 = 200
C_PUCT = 1.0
eps = 0.25   # 0 to play match games, 0.25 to train
PROF = 9  # 0 to play match games, 9 to train
eps_erase = 0.25
PROF_erase = 9

DIR_ALPHA = 1.5

from MCTS.edge import Edge     
        

# original to train the net, make a random rotation each step 
class MCTS2:
        
    def __init__(self, probs, board):
        dirich = np.random.dirichlet([DIR_ALPHA for _ in range(NUM_ACTIONS)], 1)[0]
        self.edges = [Edge((1 - eps) * probs[i] + eps * dirich[i]) for i in range(NUM_ACTIONS)]
        self.board = board


    def simulation(self, game, model):
        
        current_node = self
        current_game = copy.deepcopy(game)
        current_game.board = copy.deepcopy(self.board)

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
            finished = current_game.make_move(action_taken)
            
            # We haven't reached a leaf node yet
            if current_node.edges[action_taken].nextState != None:
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState 
                current_game.board = copy.deepcopy(current_node.board)
            
            # We have reached a leaf node, so we have to expand it
            else:
                # Winner
                if finished:
                    if len(actions) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                # Draw
                elif not current_game.is_possible_to_move():
                    total_value = 0
                # Nothing
                else:
                    # Evaluate: model(state) -> value & policy  
                    r = Rotation()
                    new_board = r.rotate_random(current_game.board)
                    current_game.board = new_board
                    state = T.tensor([current_game.get_state()]).float()
                    evaluation = model(state)[0].tolist()
                    total_value = evaluation[-1]
                    if len(actions) % 2 == 1:
                        total_value = -total_value
                    # Create next Node
                    current_node.edges[action_taken].nextState = MCTS2(evaluation[:-1], new_board)
                    
                break
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for action in actions:
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            total_value = -total_value
            

    def makeMove(self, game, model, n_games=15):

        game.board = copy.deepcopy(self.board)
        
        # Run simulations to populate our MCTS
        for _ in range(NUM_SIMULATIONS):
            self.simulation(game, model)

        # Uncomment to see the number of simulations made per action
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
            probs = [0.0 for _ in range(NUM_ACTIONS)]
            probs[action] = 1.0
        
        # We have to take care if the next state is the winning one
        state = game.get_state()
        finished = game.make_move(action) 
        
        # Go to the next node after choosing the action
        # Node + action -> next None
        try:
            if not finished and game.is_possible_to_move():               
                self.__dict__ = self.edges[action].nextState.__dict__
        except:
            print("Edges.n:", [e.n for e in self.edges])
            print("Probs:", probs)
            print("Action:", action)
            print("mcts.board", self.board)
            print("game.board", game.board)
            print("N games:", n_games)
            return 


        return action, state, probs, finished


# q_values (not as good as we spected)
class MCTS3:
        
    def __init__(self, probs):
        dirich = np.random.dirichlet([DIR_ALPHA for _ in range(NUM_ACTIONS)], 1)[0]
        self.edges = [Edge((1 - eps) * probs[i] + eps * dirich[i]) for i in range(NUM_ACTIONS)]


    def simulation(self, game, model):
        
        current_node = self
        current_game = copy.deepcopy(game)

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
            finished = current_game.make_move(action_taken)
            
            # We haven't reached a leaf node yet
            if current_node.edges[action_taken].nextState != None:
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState 
            
            # We have reached a leaf node, so we have to expand it
            else:
                # Winner
                if finished:
                    if len(actions) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                # Draw
                elif not current_game.is_possible_to_move():
                    total_value = 0
                # Nothing
                else:
                    # Evaluate: model(state) -> value & policy  
                    state = T.tensor([current_game.get_state()]).float()
                    evaluation = model(state)[0].tolist()
                    total_value = evaluation[-1]
                    if len(actions) % 2 == 1:
                        total_value = -total_value
                    # Create next Node
                    current_node.edges[action_taken].nextState = MCTS3(evaluation[:-1])
                    
                break
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for action in actions:
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            total_value = -total_value
            

    def makeMove(self, game, model):
        
        # Run simulations to populate our MCTS
        for _ in range(NUM_SIMULATIONS_3):
            self.simulation(game, model)

        # Uncomment to see the number of simulations made per action
        # print(list(map(lambda edge: edge.n, self.edges)))
        
        # Choose action depending on the simulations
        l = [(1+e.q/e.n)/2 if e.n != 0 else 0 for e in self.edges]
        s = sum(l)
        probs = [x/s for x in l]
        action = np.argmax(probs)

        # We have to take care if the next state is the winning one
        state = game.get_state()
        finished = game.make_move(action) 
        
        # Go to the next node after choosing the action
        # Node + action -> next None
        if not finished and game.is_possible_to_move():               
            self.__dict__ = self.edges[action].nextState.__dict__

        return action, state, probs, finished


# max of 3 moves then we erase 
class MCTS4:
        
    def __init__(self, probs):
        dirich = np.random.dirichlet([DIR_ALPHA for _ in range(NUM_ACTIONS)], 1)[0]
        self.edges = [Edge((1 - eps) * probs[i] + eps * dirich[i]) for i in range(NUM_ACTIONS)]


    def simulation(self, game, model):
        
        current_node = self
        current_game = copy.deepcopy(game)

        actions = [] 
        erase = current_game.count_chips() == 3 
        aux = erase or game.depth >= 6
        begin_erasing = erase
        init_depth = current_game.depth
        
        while True:

            # Find next action: a = argmax{Q(a) + U(a)} 
            Q = map(lambda edge: edge.q, current_node.edges)
            sqrt_sum = math.sqrt(sum(map(lambda edge: edge.n, current_node.edges)))
            U = [(C_PUCT * edge.p * sqrt_sum / (1 + edge.n)) for edge in current_node.edges]
            # Avoid unavailable actions 
            possible_actions = current_game.available_moves(erase=erase)
            confidence = [(q+u if i in possible_actions else -np.inf) for i,(q,u) in enumerate(zip(Q,U))]
            action_taken = np.argmax(confidence)
            actions.append(action_taken)
            
            # Update the current game
            finished = current_game.make_move(action_taken, erase=erase)

            if not aux:
                if current_game.depth == 6:
                    erase = True
                    aux = True
            else:
                erase = not erase  # Once we start erasing, we do it alternatively 
            
            # We haven't reached a leaf node yet
            if current_node.edges[action_taken].nextState != None:
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState 
            
            # We have reached a leaf node, so we have to expand it
            else:
                # Winner
                if finished:
                    if (current_game.depth - init_depth) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                        
                # Draw
                elif current_game.depth > 20 and (current_game.depth - init_depth) > 9:
                    total_value = 0
                # Nothing
                else:
                    # Evaluate: model(state) -> value & policy  
                    state = T.tensor([current_game.get_state2()]).float()
                    evaluation = model(state)[0].tolist()
                    total_value = evaluation[-1]
                    if (current_game.depth - init_depth) % 2 == 1:
                        total_value = -total_value
                    # Create next Node
                    current_node.edges[action_taken].nextState = MCTS4(evaluation[:-1])
                    
                break
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for i,action in enumerate(actions):
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            # if we pass the first 6 movements then we start erasing -> each player play twice
            if init_depth >= 6:
                if begin_erasing:
                    if i % 2 == 1:
                        total_value = -total_value
                elif i % 2 == 0:
                    total_value = -total_value
            else:
                if (init_depth + i) < 6:
                    total_value = -total_value
                elif (init_depth + i - 6) % 2 == 1:
                    total_value = -total_value
            

    def makeMove(self, game, model, n_games=15):

        # Run simulations to populate our MCTS
        for _ in range(NUM_SIMULATIONS):
            self.simulation(game, model)

        # Uncomment to see the number of simulations made per action
        # print(list(map(lambda edge: edge.n, self.edges)))

        erase = game.count_chips() == 3
        
        # Choose action depending on the simulations
        if n_games < 15:
            TAU = 1    # Choose the value of tau depending on the current depth
            sum_tot = sum([edge.n**(1/TAU) for edge in self.edges])
            probs = [(edge.n**(1/TAU)) / sum_tot for edge in self.edges]
            while True:
                action = np.random.choice(a=list(range(NUM_ACTIONS)), p=probs)
                if action in game.available_moves(erase=erase):
                    break
        else:
            action = np.argmax(list(map(lambda edge: edge.n, self.edges)))
            probs = [0.0 for _ in range(NUM_ACTIONS)]
            probs[action] = 1.0

        # We have to take care if the next state is the winning one
        state = game.get_state2()
        finished = game.make_move(action, erase=erase) 
                    
        # Go to the next node after choosing the action
        # Node + action -> next None
        try:
            if not finished:               
                self.__dict__ = self.edges[action].nextState.__dict__
        except:
            print("N_games:", n_games)
            print("Depth:", game.depth)
            print("Action:", action)
            print("Probs:", probs)
            print("Self.n", [e.n for e in self.edges])
            print("nextState == None:", self.edges[action].nextState == None)
            game.print_board()

        return action, state, probs, finished 

    def update_board(self, game, action, model):
        if self.edges[action].nextState == None:
            state = T.tensor([game.get_state2()]).float()
            eval = model(state)[0].tolist()
            self.edges[action].nextState = MCTS4(eval[:-1])
            
        self.__dict__ = self.edges[action].nextState.__dict__

    def update_board_2(self, game, action_erase, action_move, model):

        game0 = copy.deepcopy(game)
        game0.board[action_move//3][action_move%3] = 0
        game0.change_player()

        if self.edges[action_erase].nextState == None:
            state = T.tensor([game0.get_state2()]).float()
            eval = model(state)[0].tolist()
            self.edges[action_erase].nextState = MCTS4(eval[:-1])
            
        self.__dict__ = self.edges[action_erase].nextState.__dict__

        if self.edges[action_move].nextState == None:
            state = T.tensor([game.get_state2()]).float()
            eval = model(state)[0].tolist()
            self.edges[action_move].nextState = MCTS4(eval[:-1])
            
        self.__dict__ = self.edges[action_move].nextState.__dict__



# probably we will erase this class and subsitute to an other one
class MCTS_erase:
        
    def __init__(self, probs):
        dirich = np.random.dirichlet([DIR_ALPHA for _ in range(NUM_ACTIONS)], 1)[0]
        self.edges = [Edge((1 - eps_erase) * probs[i] + eps_erase * dirich[i]) for i in range(NUM_ACTIONS)]

    def move_AI(self, game, nn_move):
        state = T.tensor([game.get_state()], dtype=T.float)
        probs = nn_move(state)[0].tolist()
        action = np.argmax(probs[:-1])
        posibilities = game.available_moves()
        while action not in posibilities:
            probs[action] = 0
            action = np.argmax(probs[:-1])
        return action

    def simulation(self, board, nn, nn_move, trainning):
        
        current_node = self
        current_board = copy.deepcopy(board)

        actions = []        
        
        i = 0
        while True:

            # Find next action: a = argmax{Q(a) + U(a)} 
            Q = map(lambda edge: edge.q, current_node.edges)
            sqrt_sum = math.sqrt(sum(map(lambda edge: edge.n, current_node.edges)))
            U = [(C_PUCT * edge.p * sqrt_sum / (1 + edge.n)) for edge in current_node.edges]
            # Avoid unavailable actions 
            possible_actions = current_board.available_moves(erase=True)
            confidence = [(q+u if i in possible_actions else -np.inf) for i,(q,u) in enumerate(zip(Q,U))]
            action_taken = np.argmax(confidence)
            actions.append(action_taken)
            
            # Update the current board (erase and then move with other model)
            current_board.board[action_taken//3][action_taken%3] = 0
            action = self.move_AI(current_board, nn_move)
            win = current_board.make_move(action) 
            
            # We haven't reached a leaf node yet
            if current_node.edges[action_taken].nextState != None:
                # Move to the next state and keep traversing the MCTS
                current_node = current_node.edges[action_taken].nextState
            
            # We have reached a leaf node, so we have to expand it
            else:
                # Winner
                if win:
                    if len(actions) % 2 == 0:
                        total_value = -1
                    else:
                        total_value = 1
                # Draw
                elif i == 8:
                    total_value = 0
                # Nothing
                else:
                    # Evaluate: nn(state) -> value & policy 
                    if not trainning:
                        state = T.tensor([current_board.get_state()]).float()
                        evaluation = nn(state)[0].tolist()
                    else:
                        evaluation = nn.get_4move(current_board)
                    total_value = evaluation[-1]
                    if len(actions) % 2 == 1:
                            total_value = -total_value
                    # Create next Node
                    current_node.edges[action_taken].nextState = MCTS_erase(evaluation[:-1])
                    
                break
            i += 1
            
        # Update all the edges we have gone through during the simulation
        current_node = self
        for action in actions:
            current_node.edges[action].updateEdge(total_value)
            current_node = current_node.edges[action].nextState
            total_value = -total_value
            

    def makeMove(self, board, nn, nn_move, trainning=False):
        
        # Run simulations to populate our MCTS
        for _ in range(NUM_SIMULATIONS):
            self.simulation(board, nn, nn_move, trainning)

        # Uncomment to see the number of simulations made per action
        # print(list(map(lambda edge: edge.n, self.edges)))
        
        # Choose action depending on the simulations
        if board.depth < PROF_erase:
            TAU = 1    # Choose the value of tau depending on the current depth
            sum_tot = sum([edge.n**(1/TAU) for edge in self.edges])
            probs = [(edge.n**(1/TAU)) / sum_tot for edge in self.edges]
            action = np.random.choice(a=list(range(NUM_ACTIONS)), p=probs)
        else:
            action = np.argmax(list(map(lambda edge: edge.n, self.edges)))
            probs = [0.0 for _ in range(NUM_ACTIONS)]
            probs[action] = 1.0

        # We have to take care if the next state is the winning one
        curr_board = copy.deepcopy(board)
        curr_board.board[action//3][action%3] = 0
        action_move = self.move_AI(curr_board, nn_move)
        win = curr_board.make_move(action_move) 
        
        # Go to the next node after choosing the action
        # Node + action -> next None
        if not win and curr_board.is_possible_to_move():               
            self.__dict__ = self.edges[action].nextState.__dict__
        
        return action, copy.deepcopy(board).get_state(), probs
    
    
    def update_board(self, game, action_erase, nn):
        if self.edges[action_erase].nextState == None:
            state = T.tensor([game.get_state()]).float()
            evaluation = nn(state)[0].tolist()
            """evaluation = nn.get_4move(game)"""
            self.edges[action_erase].nextState = MCTS(evaluation[:-1])
            
        self.__dict__ = self.edges[action_erase].nextState.__dict__
    