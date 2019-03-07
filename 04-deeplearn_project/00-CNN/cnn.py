from layer_utils import *

class ThreeLayerConvNet(object):    
    """    
    A three-layer convolutional network with the following architecture:       
       conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,             
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters*H*W/4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():    
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # compute the forward pass
        a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        a2, cache2 = affine_relu_forward(a1, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)

        if y is None:    
            return scores

        # compute the backward pass
        data_loss, dscores = softmax_loss(scores, y)
        da2, dW3, db3 = affine_backward(dscores, cache3)
        da1, dW2, db2 = affine_relu_backward(da2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])

        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        return loss, grads