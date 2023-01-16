# Developed by MD-Nazmus Samin Sunbeam
import numpy as np

class Activation(object):
    def __init__(self):
        self.function = None

    def evaluate(self, x):
        raise NotImplementedError

    def derive(self, x):
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def evaluate(self, x):
        self.function = 1.0 / (1.0 + np.exp(-x))
        return self.function

    def derive(self, x):
        self.function = self.evaluate(x) * (1 - self.evaluate(x))
        return self.function

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()
    
    def evaluate(self, x):
        self.function = np.tanh(x)
        return self.function

    def derive(self, x):
        self.function = 1 - np.power(self.evaluate(x), 2)
        return self.function

class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    def evaluate(self, x):
        self.function = np.maximum(0, x)
        return self.function

    def derive(self, x):
        self.function = np.where(x > 0, 1, 0)
        return self.function

class StableSoftmax(Activation): # will need to check this implementation
    def __init__(self):
        super(StableSoftmax, self).__init__()

    def evaluate(self, x):
        self.function = np.exp(x - np.max(x)) / np.sum(np.exp(x), axis=0)
        return self.function

    def derive(self, x):
        self.function = self.evaluate(x) * (1 - self.evaluate(x))
        return self.function

class Linear(Activation):
    def __init__(self):
        super(Linear, self).__init__()

    def evaluate(self, x):
        self.function = x
        return self.functions

    def derive(self, x):
        self.function = 1
        return self.function