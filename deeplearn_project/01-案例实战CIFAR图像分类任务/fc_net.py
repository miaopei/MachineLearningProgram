from layer_utils import *
import numpy as np
class TwoLayerNet(object):   
    
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,           
                              weight_scale=1e-3, reg=0.0):    
        """    
        Initialize a new network.   
        Inputs:    
        - input_dim: An integer giving the size of the input    
        - hidden_dim: An integer giving the size of the hidden layer    
        - num_classes: An integer giving the number of classes to classify    
        - dropout: Scalar between 0 and 1 giving dropout strength.    
        - weight_scale: Scalar giving the standard deviation for random 
                        initialization of the weights.    
        - reg: Scalar giving L2 regularization strength.    
        """    
        self.params = {}    
        self.reg = reg   
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)     
        self.params['b1'] = np.zeros((1, hidden_dim))    
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)  
        self.params['b2'] = np.zeros((1, num_classes))

    def loss(self, X, y=None):    
        """   
        Compute loss and gradient for a minibatch of data.    
        Inputs:    
        - X: Array of input data of shape (N, d_1, ..., d_k)    
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].  
        Returns:   
        If y is None, then run a test-time forward pass of the model and return:    
        - scores: Array of shape (N, C) giving classification scores, where              
                  scores[i, c] is the classification score for X[i] and class c. 
        If y is not None, then run a training-time forward and backward pass and    
        return a tuple of:    
        - loss: Scalar value giving the loss   
        - grads: Dictionary with the same keys as self.params, mapping parameter             
                 names to gradients of the loss with respect to those parameters.    
        """
        scores = None
        N = X.shape[0]
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h1, cache1 = affine_relu_forward(X, W1, b1)
        out, cache2 = affine_forward(h1, W2, b2)
        scores = out              # (N,C)
        # If y is None then we are in test mode so just return scores
        if y is None:   
            return scores

        loss, grads = 0, {}
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)
        # Add the regularization gradient contribution
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads