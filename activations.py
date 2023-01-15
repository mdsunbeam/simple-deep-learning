# Developed by MD-Nazmus Samin Sunbeam
import numpy as np

class activation(object):
    def __init__(self):
        self.function = None

    def evaluate(self, x):
        raise NotImplementedError

    def derive(self, x):
        raise NotImplementedError

class sigmoid(activation):
    def __init__(self):
        super(sigmoid, self).__init__()
    
    def evaluate(self, x):
        self.function = 1.0 / (1.0 + np.exp(-x))
        return self.function

    def derive(self, x):
        self.function = self.evaluate(x) * (1 - self.evaluate(x))
        return self.function

class tanh(activation):
    def __init__(self):
        super(tanh, self).__init__()
    
    def evaluate(self, x):
        self.function = np.tanh(x)
        return self.function

    def derive(self, x):
        self.function = 1 - np.power(self.evaluate(x), 2)
        return self.function

class relu(activation):
    def __init__(self):
        super(relu, self).__init__()

    def evaluate(self, x):
        self.function = np.maximum(0, x)
        return self.function

    def derive(self, x):
        self.function = np.where(x > 0, 1, 0)
        return self.function

class stable_softmax(activation): # will need to check this implementation
    def __init__(self):
        super(stable_softmax, self).__init__()

    def evaluate(self, x):
        self.function = np.exp(x - np.max(x)) / np.sum(np.exp(x), axis=0)
        return self.function

    def derive(self, x):
        self.function = self.evaluate(x) * (1 - self.evaluate(x))
        return self.function

class linear(activation):
    def __init__(self):
        super(linear, self).__init__()

    def evaluate(self, x):
        self.function = x
        return self.functions

    def derive(self, x):
        self.function = 1
        return self.function