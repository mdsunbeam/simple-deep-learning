import numpy as np

class loss(object):
    def __init__(self):
        self.function = None
    
    def forward(self, y, y_hat):
        raise NotImplementedError

    def backward(self, y, y_hat):
        raise NotImplementedError

class mse(loss):
    def __init__(self):
        super(mse, self).__init__()

    def forward(self, y, y_hat):
        self.function = np.mean(np.power(y - y_hat, 2))
        return self.function

    def backward(self, y, y_hat):
        self.function = 2 * (y_hat - y) / y.size
        return self.function

class cross_entropy(loss):
    def __init__(self):
        super(cross_entropy, self).__init__()

    def forward(self, y, y_hat):
        self.function = -np.sum(y * np.log(y_hat))
        return self.function

    def backward(self, y, y_hat):
        self.function = -y / y_hat
        return self.function

class binary_cross_entropy(loss):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def forward(self, y, y_hat):
        self.function = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return self.function

    def backward(self, y, y_hat):
        self.function = -y / y_hat + (1 - y) / (1 - y_hat)
        return self.function

