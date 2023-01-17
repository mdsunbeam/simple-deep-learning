
class Layer(object):

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, prev_grad, *args, **kwargs):
        raise NotImplementedError

    def connection(self, prev_layer):
        raise NotImplementedError

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []

    @property
    def param_grads(self):
        return list(zip(self.params, self.grads))