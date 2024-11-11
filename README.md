# Dust Autograd Engine


[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/username/project/actions)

## Description
This repository implements a basic AutoGrad Engine purely in python without utilizing any 3rd party frameworks such as numpy. It works on a scalar level, using the class Value.


## Table of Contents
- [The _Value_ object](#the-value-object)
- [The Computational Graph and topological ordering](#the-computational-graph-and-topological-ordering)
- [The _Value.backward()_ function](#the-value.backward()-function)
- [The _backward\_func_ attribute of Value objects](#the-backward-func-attribute-of-value-objects)
- [Classification Example](#classification-example)
- [Regression Example](#regression-example)

- [License](#license)

## The _Value_ object

The Value object has the following structure
``` python
class Value(object):
    '''
    Value Object:
        arguments:
            - x: input data of type int or float
        output:
            - Value object with value = x
    '''
    def __init__(self, x, children=[], requires_grad=False):
        self.children = children
        self.backward_func = None
        self.grad = 0.0
        self.requires_grad = requires_grad

    # Returns Value object's float value
    def value(self):
        return self.val
    
    ## Operator overloading for all Python basic operators
    ## e.g.
    def __neg__(self):
        # Return a new Value instance with the negated value
        return Value(-self.val, [self])
    
    # backward first creates a list of topologically ordered elements of the computational graph
    # Then for each of this nodes calls the backward_func 
    # to fill in the respective gradients of the children
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

```

Each value object has the following attributes:

1. children: A Python List containing Value() objects that produce the parent Value() object when combined with an operator or function

E.g
<p align="center">
<img src="Images/children.jpg" alt="Image description" height="300">
</p>

2. backward_func: An object that receives a cache saving useful information from the forward pass and implements a backward(grad) call where it transmits the receive grad to the children nodes

3. grad: The accumulated gradient for the particular scalar Value() object

4. requires_grad: Whether or not the particular node requires gradient accumulation

The value object also has the following functions:

1. All operator overloading functions for all primitive python operators such as +, -, * etc.

2. value(self): Which returns the current value of the Value() object in a Python float type

3. backward(self): Which first extracts a topologically ordered list of nodes of the computational graph and then iterates over them to accumulate gradients for all the nodes of the graph

## The Computational Graph and topological ordering

Each backward() call of a given Value() object first creates a topologically ordered list of the given computational graph. It does so to prevent another Value() node from calling its backward_func when It hasn't accumulated all the gradients from all the possible paths leading to the particular node. Let us demonstrate this failure case with an example.

Assume you have constructed the following computational graph:

<p align="center">
  <img src="Images/topo.jpg" alt="Image description"  height="300">
</p>

If you don't first sort the nodes in a Topologically order way, it could be the case that the traversal of the graph in the backward() call would be of the following order:

``` python
order = [Node5, Node2, Node4, Node3, Node1]
```

However this has disastrous results since _Node2_ would call its _backward\_func_ even though gradients from Nodes 3 and 4 have not been accumulated.

Topologically ordering the above graph means that each pair of nodes from the ordered list respects its order, leading to the following list of objects:

``` python
order = [Node5, Node4, Node3, Node2, Node1]
```


## The _Value.backward()_ function

The backward() function of a Value() object computes the gradients of all the Value() object nodes in the opposite direction of the Graph.
It does so by first creating a topologically ordered list of the Nodes at hand and then traverses these nodes in the correct order and calls the backward_func of these nodes to accumulate gradients.

Namely:
``` python
topo = list(reversed(get_topological_order(self)))
```

Creates the list

``` python
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
```
Traverses the list and accumulates gradients.

In more detail:
``` python
for node in topo:
```

Iterates through the topo list

``` python
if isinstance(node, type(self)):
```

Checks if the current node is of type Value() (Could be the case that leaf nodes are simple floats or ints, this is allowed in our API)

``` python
if node.backward_func is not None:
```

Checks if the current Node has the backward_func attribute implemented (It is not implemented for leaf nodes)
``` python
grad = node.grad
grads = node.backward_func(grad)
```
Gets the current grad of the Node we are in and calculates the children grads by passing it to the backward_func()

``` python
for i, child in enumerate(node.children):
    if isinstance(child, type(self)):
        child.grad += grads[i]
```

Iterates through the children list and accumulates gradients to each child by means of an inplace += operation

``` python
else:
    continue
```

If the Value() object does not have backwawrd_func implemented (is a leaf node), do nothing.


## The _backward\_func_ attribute of Value objects

When an operator (+) or function (tanh) gets called on a Value() or Two Value() objects, the output Value() objects gets the corresponding backward_fun object as each attribute together with the associated cache

Let us give a simple example:

The Value class has the addition operator overloading implemented as:

``` python
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
```

When you then call:

``` python
c = a + b
```

The __add__(self, other) function of the _a_ object gets called.

The function then checks if the "other" object is of type Value() with:

``` python
if isinstance(other, type(self)):
```

and then creates a new Value() object which will be associated with the _c_ value as:

``` python
value = Value(self.val + other.val, [self, other])
```

The list [self, other] are the children of the new Value() object, namely self=a and other=b

Finally, backward_add() is assigned to the new object's backward_func attribute with:

``` python
value.backward_func = backward_add()
```

Let us now check on the backward_add() implementation:

``` python
class backward_add(object):
    def __init__(self, cache=None):
        self.cache = cache

    def __call__(self, grad):
        return grad, grad
```

This is a simple object, that receives a cache (In our case the + operator simply transmits the grad to its children and does not require a cache)
and a \_\_call\_\_(self, grad) method which returns a tuple of (grad, grad)

This allows us to call the backward_add object  as:

``` python
backward_add(grad)
```

## Classification Example

The notebook _test\_grad\_classification.ipynb_ implements a simple 2D training loop using the Dust Grad Module for classifying the Yin-Yang dataset. You can walk through it to see the implementation details by yourself and tune the hyperparameters to get better or worse results

The resulting Decision Boundary together with the samples is the following:

<p align="center">
  <img src="Images/topo.jpg" alt="Image description"  height="300">
</p>


## Regression Example

The notebook _test\_grad\_regression.ipynb_ implements a simple 1D training loop using the Dust Grad Module for regressing over a simple sin(x) function with the addition of noise. You can walk through it to see the implementation details by yourself and tune the hyperparameters to get better or worse results

The resulting Predicted Regression Line and the corresponding samples are as follows:

<p align="center">
  <img src="Images/topo.jpg" alt="Image description"  height="300">
</p>


