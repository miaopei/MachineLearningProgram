import numpy as np

def sgd(w, dw, config=None):    
    """    
    Performs vanilla stochastic gradient descent.    
    config format:    
    - learning_rate: Scalar learning rate.    
    """    
    if config is None: config = {}    
    config.setdefault('learning_rate', 1e-2)    
    w -= config['learning_rate'] * dw    

    return w, config

def sgd_momentum(w, dw, config=None):    
    """    
    Performs stochastic gradient descent with momentum.    
    config format:    
    - learning_rate: Scalar learning rate.    
    - momentum: Scalar between 0 and 1 giving the momentum value.                
    Setting momentum = 0 reduces to sgd.    
    - velocity: A numpy array of the same shape as w and dw used to store a moving    
    average of the gradients.   
    """   
    if config is None: config = {}    
    config.setdefault('learning_rate', 1e-2)   
    config.setdefault('momentum', 0.9)    
    v = config.get('velocity', np.zeros_like(w))    
    next_w = None    
    v = config['momentum'] * v - config['learning_rate'] * dw    
    next_w = w + v    
    config['velocity'] = v    

    return next_w, config

def rmsprop(x, dx, config=None):    
    """    
    Uses the RMSProp update rule, which uses a moving average of squared gradient    
    values to set adaptive per-parameter learning rates.    
    config format:    
    - learning_rate: Scalar learning rate.    
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared                  
    gradient cache.    
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.    
    - cache: Moving average of second moments of gradients.   
    """    
    if config is None: config = {}    
    config.setdefault('learning_rate', 1e-2)  
    config.setdefault('decay_rate', 0.99)    
    config.setdefault('epsilon', 1e-8)    
    config.setdefault('cache', np.zeros_like(x))    
    next_x = None    
    cache = config['cache']    
    decay_rate = config['decay_rate']    
    learning_rate = config['learning_rate']    
    epsilon = config['epsilon']    
    cache = decay_rate * cache + (1 - decay_rate) * (dx**2)    
    x += - learning_rate * dx / (np.sqrt(cache) + epsilon)  
    config['cache'] = cache    
    next_x = x    

    return next_x, config

def adam(x, dx, config=None):    
    """    
    Uses the Adam update rule, which incorporates moving averages of both the  
    gradient and its square and a bias correction term.    
    config format:    
    - learning_rate: Scalar learning rate.    
    - beta1: Decay rate for moving average of first moment of gradient.    
    - beta2: Decay rate for moving average of second moment of gradient.   
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.    
    - m: Moving average of gradient.    
    - v: Moving average of squared gradient.    
    - t: Iteration number.   
    """    
    if config is None: config = {}    
    config.setdefault('learning_rate', 1e-3)    
    config.setdefault('beta1', 0.9)    
    config.setdefault('beta2', 0.999)    
    config.setdefault('epsilon', 1e-8)    
    config.setdefault('m', np.zeros_like(x))    
    config.setdefault('v', np.zeros_like(x))    
    config.setdefault('t', 0)   
    next_x = None    
    m = config['m']    
    v = config['v']    
    beta1 = config['beta1']    
    beta2 = config['beta2']    
    learning_rate = config['learning_rate']    
    epsilon = config['epsilon']   
    t = config['t']    
    t += 1    
    m = beta1 * m + (1 - beta1) * dx    
    v = beta2 * v + (1 - beta2) * (dx**2)    
    m_bias = m / (1 - beta1**t)    
    v_bias = v / (1 - beta2**t)    
    x += - learning_rate * m_bias / (np.sqrt(v_bias) + epsilon)    
    next_x = x    
    config['m'] = m    
    config['v'] = v    
    config['t'] = t    

    return next_x, config