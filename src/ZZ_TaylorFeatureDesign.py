import numpy as np

import jax.numpy as jnp
from jax import grad, jacfwd
from jax import random

# A bunch of functions that might be promising implemented as default. 
# You can always used the dedicated methods of TFD to check how the learned features compare with other features

def logvar(a):
    return jnp.log(jnp.var(a))

def logsumsquare(a):
    return jnp.log(jnp.sum(jnp.power(a,2)))

def expvar(a):
    return jnp.exp(jnp.var(a))

def rone_to_rone(id_): 
    '''Returns a function pointing from R1 to R1, based on id
    '''
    if id_==0: 
        return lambda x: jnp.log(x)
    if id_==1: 
        return lambda x: jnp.exp(x)
    if id_==2: 
        return lambda x: jnp.power(x, 2)
    # if id_==3:
        # Written as the square root of the absolute value
    #    return lambda x: jnp.sqrt(jnp.abs(x))
    # if id_==4: 
    #    return lambda x: jnp.power(x, -1)
    else:
        raise ValueError('Not impleted, check allowed range of id')

def moment(X, power): 
    '''rewriting the sample moment without prefactor! 1/n
    operating on a single row of a matrix
    '''
    X_tilde = jnp.array(X) - jnp.mean(X)
    if len(X.shape)==2:
        shape = X.shape[1]
    else:
        shape = X.shape[0]
    return jnp.sum(jnp.power(X_tilde, power))/shape

def rm_to_rone(id_): 
    '''Returns a function pointing from Rm to R1, based on id
    '''
    if id_==0: 
        return lambda x: jnp.mean(x)
    if id_==1: 
        return lambda x: jnp.var(x)
    if id_==2: 
        return lambda x: moment(x,3)/((moment(x,2))**(3/2))
    if id_==3:
        return lambda x: moment(x,4)/(moment(x,2)**2)-3
    if id_==4:
        return lambda x: jnp.max(jnp.abs(x))
    if id_==5:
        return lambda x: jnp.sum(jnp.power(x,2))
    if id_==6:
        return lambda x: jnp.sum(jnp.power(x,3))
    if id_==7:
        return lambda x: jnp.sum(jnp.power(x,4))
    if id_==8:
        return lambda x: jnp.sum(jnp.power(x,5))
    if id_==9:
        return lambda x: jnp.sum(jnp.exp(x))
    if id_==10:
        return lambda x: jnp.sum(jnp.log(jnp.abs(x)))
    if id_==11:
        return lambda x: jnp.sum(jnp.sqrt(jnp.abs(x)))
    if id_==12:
        return lambda x: jnp.sum(jnp.power(x, -1))
    else:
        raise ValueError('Not impleted, check allowed range of id')
        
rm_to_rone_names = ['mean(X)', 'var(X)', 'skewness(X)', 'kurtosis(X)', 'max(abs(X))',
                    'sum(power(X,2))', 'sum(power(X,3))', 'sum(power(X,4))',
                    'sum(power(X,5))', 'sum(exp(X))', 'sum(log(|X|))', 
                    'sum(sqrt(|X|))', 'sum(1/X)']




class TaylorFeatureDesigner(object):
    def __init__(self, X:np.array, y:np.array, pls:bool, max_comp_pls:int, rr:bool, reg_coef_rr:np.array) -> None:
        self.X = X
        self.y = y
        pass


    def evaluate_f(self, function, **kwargs) -> None: 
        pass