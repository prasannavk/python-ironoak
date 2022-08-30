'''

'''
import numpy as np

def mycomposeRt(R,t):
    '''
    Compose 3x3 R and 3x1 t into a 4x4 transfrom T
    Args:
        R (): 3x3 numpy array
        t (): 3x1 numpy array
    Returns:
        4x4 T(R,t)
    '''
    T = np.zeros((4, 4), dtype=float)
    T[0:3,0:3] = R
    T[0:3,3] = t
    T[3,3] =  1.0
    return T

def myinvertT(T):
    '''
    Invert 4x4 numpy transform matrix T
    Args:
        T (): 4x4 float numpy Transform matrix
    Returns: T^-1
    '''
    return np.linalg.inv(T)

def my3to4pt(p):
    '''
    Convert a 3x1 numpy 3d point to homogeneous 4x1 vector
    Args:
        p (): 3x1 numpy 3D point
    Returns: p(x,y,z) -> p(x,y,z,1)
    '''
    four1s = np.ones(4, dtype=float)
    four1s[0:3] = p
    return four1s
