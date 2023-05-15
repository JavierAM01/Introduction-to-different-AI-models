
class Edge:
    
    def __init__(self, prob):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = prob
        self.nextState = None
    
    def updateEdge(self, value):
        self.n = self.n + 1
        self.w = self.w + value
        self.q = self.w / self.n   