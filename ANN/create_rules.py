from pickle_files import load, save
import numpy as np



def create_rules():

    # load data
    global codes, qtable
    codes = load("RL/models/codes.pkl")
    qtable = load("RL/models/q_100k_e1.pkl")

    # create rules
    rules = dict()
    for code in codes.keys():
        a = get_action(code)
        if a != -1:
            rules[code] = a

    # save rules
    save("ANN/models/rules.pkl", rules)

def valid(code, action):
    return (code[action] == 0)

def get_action(code):
    q_values = qtable[codes[code]]
    posibilities = np.array([a for a in range(9) if valid(code, a)])
    if len(posibilities) == 0:
        return -1
    weights = q_values[posibilities]
    # we want to return de index of the best q-value
    i = np.argmax(weights)
    action = posibilities[i]
    return action