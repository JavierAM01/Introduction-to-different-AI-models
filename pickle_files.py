import pickle

def load(path):
    f = open(path, "rb")
    X = pickle.load(f)
    f.close()
    return X

def save(path, X):
    f = open(path, "wb")
    pickle.dump(X, f)
    f.close()