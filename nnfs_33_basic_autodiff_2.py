

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
    def __init__(self, vs = [], typ:str = ''):
        self.vs = vs 
        self.typ = typ
        self.name = self.make_name(typ)
        self.result = Variable(None, made_from = self.name)
        for v in self.vs:
            v.goes_to.append(self)
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
    def __init__(self, vs):
        super().__init__(vs, 'add')
    def forward(self):
        self.result.value = self.vs[0].value + self.vs[1].value
    def backward(self):
        self.vs[0].grad += self.result.grad
        self.vs[1].grad += self.result.grad

class _Mul(Op):
    def __init__(self, vs):
        super().__init__(vs, 'mul')
    def forward(self):
        self.result.value = self.vs[0].value * self.vs[1].value
    def backward(self):
        self.vs[0].grad += self.result.grad * self.vs[1].value
        self.vs[1].grad += self.result.grad * self.vs[0].value

def Mul(v1, v2):
    op = _Mul([v1, v2])
    return op.result

def Add(v1, v2):
    op = _Add([v1, v2])
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
        for v in var.made_from.vs:
            p(v, depth + 1)
p(b)



_g = []
def create_graph(var): # topological sort 
    op = var.made_from
    if op == 'input':
        if var not in _g: # so we don't have duplicates 
            _g.append(var)
    else:
        for i in op.vs:
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