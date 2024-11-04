"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

import numpy as np


def mul(a, b):
    return a * b

def id(x):
    return x

def add(a, b):
    return a + b

def neg(x):
    return -x

def lt(a, b):
    if a < b:
        return 1.
    return 0.

def eq(a, b):
    if a == b:
        return 1.
    return 0.

def max(a, b):
    if a > b:
        return a
    return b

def is_close(a, b):
    if abs(a - b) < 1e-2:
        return 1.
    else:
        return 0.

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + math.exp(-x))

def relu(x):
    if x > 0:
        return x
    return 0.

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def inv(x):
    return 1 / x


def log_back(x, d_out):
    """
    Computes the derivative of the natural logarithm with respect to x,
    and then multiplies it by d_out.
    """
    return (1 / x) * d_out


def inv_back(x, d_out):
    """
    Computes the derivative of the reciprocal function (1/x) with respect to x,
    and then multiplies it by d_out.
    """
    return (-1 / (x ** 2)) * d_out


def relu_back(x, d_out):
    """
    Computes the derivative of the ReLU function with respect to x,
    and then multiplies it by d_out.
    """
    relu_derivative = np.where(x > 0, 1, 0)  # Derivative of ReLU: 1 if x > 0, else 0
    return relu_derivative * d_out

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(itr, f):
    for el in itr:
        yield f(el)


def zipWith(it1, it2, f):
    for el1, el2 in zip(it1, it2):
        yield f(el1, el2)


def reduce(f, itr, initial=None):
    """
    Reduces an iterable to a single value using a given function.
    """
    iterator = iter(itr)

    if initial is None:
        try:
            result = next(iterator)
        except StopIteration:
            return 0
    else:
        result = initial

    for el in iterator:
        result = f(result, el)

    return result


def negList(lst):
    return list(map(lst, neg))


def addLists(l1, l2):
    return list(zipWith(l1, l2, add))

def sum(lst):
    return reduce(add, lst)

def prod(lst):
    return reduce(mul, lst)
