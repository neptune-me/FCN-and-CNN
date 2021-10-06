from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b1'] = np.zeros(hidden_dim) #这里用zeros比zeros_like更方便
        self.params['b2'] = np.zeros(num_classes)		
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


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
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        a2, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(a2, self.params['W2'], self.params['b2'])
		   #因为前面已经有前向传播和后向传播的函数了，直接调用就行
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_without_reg, dscores = softmax_loss(scores, y) #直接用定义的softmax_loss计算未加正则项的损失
        loss = loss_without_reg+0.5*self.reg*(np.sum(self.params['W1']**2)+np.sum(self.params['W2']**2))
		                            #有几个W就需要往正则项中加几项W^2
        da2, grads['W2'], grads['b2'] = affine_backward(dscores, cache2) #注意cache中元素的调用
        grads['W2'] += self.reg*cache2[1] #这里只加一倍的reg*W2得益于上面定义的损失函数正则项有0.5
        dx, grads['W1'], grads['b1'] = affine_relu_backward(da2, cache1)
        grads['W1'] += self.reg*cache1[0][1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        shape1 = input_dim
        for i, shape2 in enumerate(hidden_dims): #shape2是用来循环每一个隐藏层神经元数量的，若有6个隐藏层，下面最多循环到W6
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(shape1, shape2) #从W1开始，逐个初始化权重矩阵
            self.params['b'+str(i+1)] = np.zeros(shape2) #从b1开始，逐个初始化偏置项
            if self.normalization == 'batchnorm': #如果有bn那么就连gamma,beta一块也初始化
                self.params['gamma'+str(i+1)] = np.ones(shape2)
                self.params['beta'+str(i+1)] = np.zeros(shape2)
            shape1 = shape2 #下一个权重矩阵的维度为shape2*下一个shape2
        self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(shape1, num_classes) #假如总共有6个隐藏层，那么这里就是W7
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        h, cache1, cache2, cache3, cache4, bn, out = {}, {}, {}, {}, {}, {},{}
		   #这里的cache‘n’分别指的是affine,bn,relu,dropout输出的cache,都是字典，存储着各个层的缓存
        out[0] = X #存储每一层的out，按照逻辑，X就是out0[0]
 
        # Forward pass: 计算得分矩阵
        for i in range(self.num_layers - 1): #减去1以后变成隐藏层的数量，这个循环一层一层地计算前向传播的数据
            # 得到每一层的参数
            w, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)] #每一层先把W,b赋值
            if self.normalization == 'batchnorm':
                gamma, beta = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)] #如果使用BN那么就将gamma,beta也赋值
                h[i], cache1[i] = affine_forward(out[i], w, b)  #这三行代码前向传播数据
                bn[i], cache2[i] = batchnorm_forward(h[i], gamma, beta, self.bn_params[i])
                out[i + 1], cache3[i] = relu_forward(bn[i])
                if self.use_dropout: #dropout操作是在准备输出时才进行的
                    out[i+1], cache4[i] = dropout_forward(out[i+1] , self.dropout_param)
            else: #如果不使用BN那么前向传播就只有一行，直接调用affine_relu_forward函数
                out[i + 1], cache3[i] = affine_relu_forward(out[i], w, b)
                if self.use_dropout:
                    out[i + 1], cache4[i] = dropout_forward(out[i + 1], self.dropout_param)
 
        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)] #这是最后一层的W,b
        scores, cache = affine_forward(out[self.num_layers - 1], W, b) #scores在最后一层才计算，传给最终输出
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(self.num_layers): #将每一层的W都计算一个正则化项加在最终梯度中
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(i + 1)] * self.params['W' + str(i + 1)])
        loss = data_loss + reg_loss
        # Backward pass: 算梯度
        dout, dbn, dh, ddrop = {}, {}, {}, {}
        t = self.num_layers - 1 #隐藏层数量
		   #这一行是计算最后一层的dout
        dout[t], grads['W' + str(t + 1)], grads['b' + str(t + 1)] = affine_backward(dscores, cache) #最后一层只有affine，用这个计算这一层的梯度，其中dscores指的是softmax计算的梯度     
        #开始计算隐藏层的dout,层数的计法与正向传播是一样的
        for i in range(t):
            if self.normalization == 'batchnorm':
                if self.use_dropout: #因为是反向传播，所以这里的dropout要写在最前面
                    dout[t - i] = dropout_backward(dout[t-i], cache4[t-1-i])
               #这三行用来反向计算各个梯度和传播微分值
                dbn[t - 1 - i] = relu_backward(dout[t - i], cache3[t - 1 - i])
                dh[t - 1 - i], grads['gamma' + str(t - i)], grads['beta' + str(t - i)] = batchnorm_backward(dbn[t - 1 - i],cache2[t - 1 - i])                                                                                           
                dout[t - 1 - i], grads['W' + str(t - i)], grads['b' + str(t - i)] = affine_backward(dh[t - 1 - i],cache1[t - 1 - i])                                                                                                 
            else:
                if self.use_dropout:
                    dout[t - i] = dropout_backward(dout[t - i], cache4[t - 1 - i])
                dout[t - 1 - i], grads['W' + str(t - i)], grads['b' + str(t - i)] = affine_relu_backward(dout[t - i],cache3[t - 1 - i])
        for i in range(self.num_layers):# 梯度计算也增加正则化项，只有W的梯度要加，因为只用了W计算正则化项
            grads['W' + str(i + 1)] += self.reg * self.params['W' + str(i + 1)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
