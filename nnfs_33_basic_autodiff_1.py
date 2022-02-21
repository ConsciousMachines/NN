

class Variable:
    c = 0
    def __init__(self, value, name:str = '', made_from = 'input'):
        if name == '': name = self.make_name()
        self.grad = 0
        self.value = value 
        self.name = name
        self.goes_to = []
        self.made_from = made_from
    @classmethod
    def make_name(cls):
        cls.c += 1
        return f'var_{cls.c}'
    def __repr__(self):
        return f'{self.name}'

class Op:
    c = 0
    def __init__(self, v1, v2, typ:str):
        self.v1 = v1 
        self.v2 = v2 
        self.typ = typ
        self.name = self.make_name(typ)
        self.result = Variable(None, made_from = self.name)
        v1.goes_to.append(self)
        v2.goes_to.append(self)
        self.result.made_from = self
    @classmethod
    def make_name(cls, name):
        cls.c += 1
        return f'{name}_{cls.c}'
    def forward():
        raise NotImplementedError()
    def backward():
        raise NotImplementedError()
    def __repr__(self):
        return f'op:{self.name}'

class _Add(Op):
    def __init__(self, v1, v2):
        super().__init__(v1, v2, 'add')
    def forward(self):
        self.result.value = self.v1.value + self.v2.value
    def backward(self):
        self.v1.grad += self.result.grad
        self.v2.grad += self.result.grad

class _Mul(Op):
    def __init__(self, v1, v2):
        super().__init__(v1, v2, 'mul')
    def forward(self):
        self.result.value = self.v1.value * self.v2.value
    def backward(self):
        self.v1.grad += self.result.grad * self.v2.value
        self.v2.grad += self.result.grad * self.v1.value

def Mul(v1, v2):
    op = _Mul(v1, v2)
    return op.result

def Add(v1, v2):
    op = _Add(v1, v2)
    return op.result










a = Variable(1,name='a')
two = Variable(4,name='two')
three = Variable(6,name='three')

a_times_two = Mul(a, two)
a_times_three = Mul(a, three)
b = Add(a, Add(Add(a_times_two, a_times_three), a))


'''
a = 1
two = 2
three = 3
a_times_two = a * two 
a_times_three = a * three 
b = a_times_two + a_times_three

b = a*2 + a*3
'''


def p(var, depth = 0):
    line = '\t' * depth + repr(var.made_from) + ' -> ' + repr(var)
    print(line)
    if var.made_from != 'input':
        p(var.made_from.v1, depth + 1)
        p(var.made_from.v2, depth + 1)
p(b)



_g = []
def create_graph(var): # topological sort 
    op = var.made_from
    if op == 'input':
        if var not in _g: # so we don't have duplicates 
            _g.append(var)
    else:
        for i in [op.v1, op.v2]:
            create_graph(i)
        if op not in _g:
            _g.append(op)
create_graph(b)
_g


# forward pass 
for i in _g:
    if issubclass(i.__class__, Op):
        print(i)
        i.forward()

# backward pass 
b.grad = 1
for i in reversed(_g):
    if issubclass(i.__class__, Op):
        print(i)
        i.backward()

a.grad