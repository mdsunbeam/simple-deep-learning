import numpy as np

class optimizer(object):
    def __init__(self, lr=3e-4, clip=-1.0, decay=0.0, lr_min = 0e0, lr_max=np.inf):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iterations = 0

    def update(self, params, grads):
        self.iterations += 1
        self.lr *= 1.0 / (1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

class sgd(optimizer): # might need to debug this implementation
    def __init__(self, lr=3e-4, clip=-1.0, decay=0.0, lr_min = 0e0, lr_max=np.inf):
        super(sgd, self).__init__(lr, clip, decay, lr_min, lr_max)

    def update(self, params, grads):
        super(sgd, self).update(params, grads)
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class momentum(optimizer): # might need to debug this implementations
    def __init__(self, lr=3e-4, clip=-1.0, decay=0.0, lr_min = 0e0, lr_max=np.inf, beta=0.9):
        super(momentum, self).__init__(lr, clip, decay, lr_min, lr_max)
        self.beta = beta
        self.v = None

    def update(self, params, grads):
        super(momentum, self).update(params, grads)
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grad
            param -= self.lr * self.v[i]

class adam(optimizer): # might need to debug this implementation
    def __init__(self, lr=3e-4, clip=-1.0, decay=0.0, lr_min = 0e0, lr_max=np.inf, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(adam, self).__init__(lr, clip, decay, lr_min, lr_max)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, params, grads):
        super(adam, self).update(params, grads)
        if self.m is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * np.power(grad, 2)
            m_hat = self.m[i] / (1 - np.power(self.beta_1, self.iterations))
            v_hat = self.v[i] / (1 - np.power(self.beta_2, self.iterations))
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
