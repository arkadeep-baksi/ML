### For Neural Networks the expressions for finding the derivatives is massive, so we need some kind of a datastructure for this

# Python Magic methods are the methods starting and ending with double underscores ‘__’. They are defined by built-in classes in Python and commonly used for operator overloading.

# They are also called Dunder methods, Dunder here means “Double Under (Underscores)”.

class Value:
    """ Stores a single scalar value and its gradient """
    def __init__(self, data, _children=(), _op="", label=""):

        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):

        return f"Value(data={self.data})"

    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        ## Backpropagation value in case of addition operation
        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data - other.data, (self, other), '-')

        ## Backpropagation value in case of subtraction operation
        def _backward():
            self.grad += output.grad
            other.grad += -1.0 * output.grad

        output._backward = _backward
        return output

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __neg__(self):

        return self * -1

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        ## Backpropagation value in case of multiplication operation
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __div__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data / other.data, (self, other), "/")

        ## Backpropagation value in case of division operation
        def _backward():
            self.grad += (1 / other.data) * output.grad
            other.grad += -1.0 * self.data * (1 / other.data ** 2) * output.grad

        output._backward = _backward
        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output._backward = _backward
        return output

    def exp(self):

        x = self.data
        exp = math.exp(x)

        t = Value(exp, (self,), 'exp')

        ## Backpropagation value in case of Activation Function operation
        def _backward():
            self.grad += (t.grad) * exp

        t._backward = _backward
        return t

    def tanh(self):

        x = self.data
        tanh_x = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        t = Value(tanh_x, (self,), 'tanh')

        ## Backpropagation value in case of Activation Function operation
        def _backward():
            self.grad += (t.grad) * (1 - tanh_x ** 2)

        t._backward = _backward
        return t

    def relu(self):

        x = self.data
        t = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        ## Backpropagation value in case of Activation Function operation
        def _backward():
            self.grad += (self.data < 0) * t.grad

        t._backward = _backward
        return t

    def backward(self):

        topo, visited = [], set()

        # Topological sort to get the order of the backpropagation
        def build_topo(v):

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

