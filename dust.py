from typing import Any, Union
import random
import math

# -- Value object
# -- Should handle operations with the following data types:
#       1) Value-Value
#       2) Value-int
#       3) Value-float
class Value(object):
    '''
    Value Object:
        arguments:
            - x: input data of type int or float
        output:
            - Value object with value = x
    '''
    def __init__(self, x:Union[float, int, str], children=[], requires_grad=False):
        self.children = children
        self.backward_func = None
        self.grad = 0.0
        self.requires_grad = requires_grad
        self.valid_dtypes = [float, int, Value, str]
        assert type(x) in self.valid_dtypes, "Invalid input dtype {}. Should be in: {}".format(type(x), [repr(dtype) for dtype in self.valid_dtypes])
        if type(x) in [int, str]:
            try:
                x = float(x)
            except:
                raise Exception("str object cannot be converted to either float or int")
            self.val = x
        elif type(x) == Value:
            self.val = x.val
            del x
        else:
            self.val = x

    def value(self):
        return self.val
    
    def __repr__(self):
        return "Value(data={})".format(self.val)

    def __str__(self):
        return "Value(data={})".format(self.val)
    
    def __neg__(self):
        # Return a new Value instance with the negated value
        return Value(-self.val, [self])
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val + other.val, [self, other])
        else:
            try: 
                value = Value(self.val + other, [self, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        
        value.backward_func = backward_add()
        return value

    def __radd__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val - other.val, [self, other])
        else:
            try: 
                value = Value(self.val - other, [self, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        
        value.backward_func = backward_sub()
        return value

    def __rsub__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__sub__(other)
                            
    def __mul__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val * other.val, [self, other])
            value.backward_func = backward_mul(cache=[self.val, other.val])
        else:
            try: 
                value = Value(self.val * other, [self, other])
                value.backward_func = backward_mul(cache=[self.val, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        return value
    
    def __rmul__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val / other.val, [self, other])
            value.backward_func = backward_div(cache=[self.val, other.val])
        else:
            try: 
                value = Value(self.val / other, [self, other])
                value.backward_func = backward_div(cache=[self.val, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        return value

    def __rtruediv__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__div__(other)
    
    def backward(self):
        self.grad = 1.0
        # -- Get topological ordering with respect to self
        topo = list(reversed(get_topological_order(self)))
        for node in topo:
            if isinstance(node, type(self)):
                if node.backward_func is not None:
                    grad = node.grad
                    grads = node.backward_func(grad)
                    for i, child in enumerate(node.children):
                        if isinstance(child, type(self)):
                            child.grad += grads[i]
                else:
                    continue

        



### Backward Function Objects
class backward_add(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return grad, grad


class backward_sub(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return grad, -grad
    

class backward_mul(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return grad * self.cache[1], grad * self.cache[0]
    

class backward_div(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return grad / self.cache[1], - grad * self.cache[0] / (self.cache[1] * self.cache[1])
    

# In Backward_tanh, cache is the output of the forward pass
class backward_tanh(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return [grad * (1 - self.cache * self.cache)]


class backward_CE(object):
    def __init__(self, cache=None):
        self.cache = cache
        
    def __call__(self, grad):
        eps = 0.0001
        return [ -0.5 * self.cache[0] / (self.cache[1] + eps) + 0.5 * (1 - self.cache[0]) / (1 - self.cache[1] + eps)]
    

def tanh(x):
    if not isinstance(x, Value):
        return tanh_func(x)
    else:
        out = Value(tanh_func(x.val), [x])
        out.backward_func = backward_tanh(out.val)
    return out

def cross_entropy(p, q):
    if not isinstance(p, Value):
        out = Value(cross_entropy_func(p, q.val), [q])
        out.backward_func = backward_CE([p, q.val])
    else:
        out = Value(cross_entropy_func(p.val, q.val), [q])
        out.backward_func = backward_CE([p.val, q.val])
    return out

def cross_entropy_func(p, q):
    eps = 0.0001
    bce = -0.5 * p * math.log2(q + eps) - 0.5 * (1- p) * math.log2(1 - q + eps)

    return bce

## -- Topological ordering functionality
def topo_order_recursive(root, topo, visited):
    if root not in visited:
        visited.add(root)
        if isinstance(root, Value):
            for node in root.children:
                topo_order_recursive(node, topo, visited)
        topo.append(root)

def get_topological_order(root):
    visited = set()
    topo = []
    topo_order_recursive(root, topo, visited)
    return topo



def tanh_func(x):
    ex = math.pow(math.e, x)
    e_min_x = math.pow(math.e, -x)
    return (ex - e_min_x) / (ex + e_min_x)


class Neuron(object):
    def __init__(self, d_in):
        self.d_in = d_in
        self.weights = [Value(random.uniform(-1, 1)) for i in range(self.d_in)]
        self.bias = Value(0.0)
        self.params = [weight for weight in self.weights] + [self.bias]

    def __call__(self, x):
        assert len(x) == len(self.weights)
        out = sum([x[i] * self.weights[i] for i in range(self.d_in)]) + self.bias
        out = tanh(out)

        return out
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

    def get_number_of_params(self):
        return len(self.params)
    

class Layer(object):
    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        self.neurons = [Neuron(d_in) for i in range(d_out)]
        self.params = [param for neuron in self.neurons for param in neuron.params]
    def __call__(self, x):
        assert len(x) == self.d_in
        out = [neuron(x) for neuron in self.neurons]

        return out
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0
    def get_number_of_params(self):
        return len(self.params)

class MLP(object):
    def __init__(self, d_in, d_out, intermediate):
        self.d_in = d_in
        self.d_out = d_out
        self.depth = len(intermediate) + 2
        self.intermediate = intermediate
        self.layers = [Layer(self.d_in, self.intermediate[0])]
        for i in range(len(intermediate) - 1):
            self.layers.append(Layer(self.intermediate[i], self.intermediate[i+1]))
        self.layers.append(Layer(self.intermediate[-1], self.d_out))
        self.params = [param for layer in self.layers for param in layer.params]

    def __call__(self, x):
        out = x
        assert len(x) == self.d_in
        for layer in self.layers:
            out = layer(out)

        return out
    
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0
    def get_number_of_params(self):
        return len(self.params)


class BaseOptimizer(object):
    def __init__(self, params, lr=0.001, weight_decay=None, decay_type=2):
        self.params = params
        self.lr = lr
        self.decay_type = decay_type
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if self.weight_decay:
                if self.decay_type == 2:
                    param.val = param.val - self.lr * param.grad - self.weight_decay * (2 * param.val)
                else:
                    param.val = param.val - self.lr * param.grad 
                    l1 = -1 if param.val > 0 else -1
                    param.val -= self.weight_decay * l1
            else:
                param.val = param.val - self.lr * param.grad 

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

