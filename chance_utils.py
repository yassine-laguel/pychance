

import numpy as np
import math

### Returns a random vector v, uniformly distributed belonging to the sphere.
def random_v(dimension):
    v = np.random.normal(0.0, 1, dimension)
    v = v/ np.linalg.norm(v)
    v = np.matrix(v)
    v = np.transpose(v)
    return v

### Returns a matrix M of dimension m and of order n, ie M^n = Id
def generate_n_order_rotation(dimension, n=20):
    angle = 2 * math.pi / n
    res = np.matlib.zeros((dimension, dimension))

    res[0, 0] = math.cos(angle)
    res[0, 1] = -1 * math.sin(angle)
    res[1, 0] = math.sin(angle)
    res[1, 1] = math.cos(angle)

    return res


### Only useful for the case m = 2

def make_vector(x,y):
    u = [x,y]
    u = np.matrix(u)
    u = np.transpose(u)
    return u

