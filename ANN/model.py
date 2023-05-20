import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
from pickle_files import load
import matplotlib.pyplot as plt


device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def save_model(model, path):
    T.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(T.load(path))


class ANN_Model(nn.Module):

    def __init__(self, path=None):
        super(ANN_Model, self).__init__()

        # architecture
        self.l1 = nn.Linear(9,16)  # (16 = 8*2) ---> 8 posibles ways to win | 2 players
        self.l2 = nn.Linear(16,16)
        self.l3 = nn.Linear(16,9)  # 9 posible actions (n_outputs)

        # load model if != None
        if path != None:
            load_model(self, path)

        # to device
        self.to(device)

    def forward(self, x):
        for layer in [self.l1, self.l2, self.l3]:
            x = F.relu(layer(x))
        return F.softmax(x, dim=1)

    def move(self, game, only_action=False):
        x = T.tensor(game.board.reshape((1,9)), dtype=T.float32).to(device)
        y = self.forward(x)[0]
        map = T.tensor([(1 if game.valid(a) else -1) for a in range(9)]).to(device)
        y = y*map
        a = int(T.argmax(y))
        if only_action:
            return a
        game.insert(a)


class ANN_trainner:

    def __init__(self, model, lr=0.001):

        self.model = model
        self.rules = load("Minimax/models/rules.pkl")
        self.n = len(self.rules)  # 5920  -> a bit less than 6064 because we avoid the full boards

        # optimizer & loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss      = nn.MSELoss()

    def create_trainning_data(self, length):
        def aux(action):
            return [(1 if a == action else 0) for a in range(9)]
        index = np.random.randint(0,self.n, length)
        all = np.array(list(self.rules.keys()))
        _X = np.array(all[index])
        Y = T.tensor([aux(self.rules[tuple(x)]) for x in _X], dtype=T.float32).to(device)
        X = T.tensor(_X, dtype=T.float32).to(device)
        return X, Y
        
    def fit(self, length, batch_size, epochs, save_path=None):

        X, Y = self.create_trainning_data(length)

        n = len(X)
        losses = []
        
        for epoch in range(epochs):
            for i in range(0,n,batch_size):

                # get sample
                X_batch = X[i:i+batch_size] if i+batch_size <= n else X[i:]
                Y_batch = Y[i:i+batch_size] if i+batch_size <= n else Y[i:]
                
                # get predictions
                Z_batch = self.model.forward(X_batch)
                
                # train
                self.optimizer.zero_grad()
                loss = self.loss(Z_batch, Y_batch).to(device)
                losses.append(loss)
                loss.backward()
                self.optimizer.step()
            print("[{}] loss: {:.4f} | mean loss: {:.4f}".format(epoch, losses[-1], sum(losses)/len(losses)))

        losses = T.tensor(losses).tolist()

        if save_path != None:
            save_model(self.model, save_path)
            self.plot_loss(losses, save_path)

        return losses

    def plot_loss(self, losses, save_path=None):
        plt.title("Trainning loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        k = len(losses) // 100
        plt.plot(losses[::k])
        if save_path != None:
            plt.savefig(save_path[:-4] + ".png")
        plt.show()