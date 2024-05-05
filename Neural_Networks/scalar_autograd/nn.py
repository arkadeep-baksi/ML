## Modules for the Scalar Autograd Library
## Some functionalities mimic the API's proved by Pytorch

class Module:

  def parameters(self):

    return []

  def zero_grad(self):

    for p in self.parameters():
      p.grad = 0.0

class Neuron(Module):
  """
  Class that represents a neuron
  """

  def __init__(self, n_inp : int, nonlin : bool = True, label : str = ""):
    self.w = [Value(random.uniform(-1,1), label = label if label else f"w_{i}" ) for i in range(n_inp)]
    self.b = Value(0, label = "b")
    self.nonlin = nonlin

  def __call__(self, x : list, activation : str = 'tanh'):

    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    act.label = "Activation"
    return act.tanh() if activation == 'tanh' else act.relu()

  def parameters(self):

    return self.w + [self.b]

  def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
  """
  Class that represents a Layer in a Multi-Layer Perceptron
  """

  def __init__(self, n_inp : int, n_out : int, **kwargs):

    self.neurons = [Neuron(n_inp, **kwargs) for _ in range(n_out)]

  def __call__(self, x : list):

    out = [neuron(x) for neuron in self.neurons]
    return out[0] if len(out)==1 else out

  def parameters(self):

    return [params for neuron in self.neurons for params in neuron.parameters()]

  def __repr__(self):

    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
  """
  Class that represents a Multi-Layer Perceptron
  A MLP can be thought of n number of Layers where each Layer contain m Neurons
  """
  def __init__(self, n_inp : int, n_outs : list, **kargs):

    layer_sz = [n_inp]+n_outs
    self.layers = [Layer(layer_sz[i], layer_sz[i+1], label = f"Layer_{i}") for i in range(len(layer_sz)-1)]

  def __call__(self, x : list[list]):

    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]

  def __repr__(self):
      return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
