# from deep learning 3 (japanese)
import numpy as np


class Variable:
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data
        self.grad = None
        self.creator = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        self.input = input
        self.output = Variable(self.forward(input.data))
        self.output.set_creator(self)
        return self.output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return gy * 2 * self.input.data






#### MY RETARDED VERSION #################################################################
#### MY RETARDED VERSION #################################################################

class Variable:
    def __init__(self, data):
        self.data = data
        self.primal = None
        self.creator = None


class Op:
    def __call__(self, in1, in2 = None):
        self.in1 = in1
        self.in2 = in2
        self.output = Variable(self.forward())
        self.output.creator = self
        return self.output
    def forward():
        raise NotImplementedError
    def backward(g):
        raise NotImplementedError

class Add(Op):
    def forward(self):
        return self.in1.data + self.in2.data
    def backward(self, g):
        return [g ,g]

class Mul(Op):
    def forward(self):
        return self.in1.data * self.in2.data
    def backward(self, g):
        return [g*self.in2.data, g*self.in1.data]

class Square(Op):
    def forward(self):
        return self.in1.data * self.in1.data
    def backward(self, g):
        return g * 2 * self.in1.data

class Exp(Op):
    def forward(self):
        return np.exp(self.in1.data)
    def backward(self, g):
        return g * np.exp(self.in1.data)

a = Variable(2)
b = Variable(3)
c = Variable(4)
d = Square(a)
