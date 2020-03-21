from builtins import range
import numpy as np

####x
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    D = w.shape[0]
    data = x.reshape(N, D)
    out = data.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


####x
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshape = x.reshape(x.shape[0], -1)
    dx_reshape = np.dot(dout, w.T)
    dx = dx_reshape.reshape(x.shape)
    dw = np.dot(x_reshape.T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


####x
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


####x
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout
    # print(dout)
    # print(x)
    dx[x<=0]=0
    # print(dx)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx





###x
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ## OUR DROPOUT PROB IS TO INCLUDE THE NODE. HENCE THIS
        mask = np.random.rand(*x.shape) < p 
        x *= mask 
        out = x/p
        # print(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

		out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


###x
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        
        ## OUR DROPOUT PROB IS TO INCLUDE THE NODE. HENCE THIS
       	dx = dout * mask
        p = dropout_param['p']
        # print(p)
        dx /= p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

###x
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (N, Chan, Height, Width) = x.shape
    (F, Chan, HH, WW) = w.shape
    H_out = (int)(1 + (Height + 2 * conv_param['pad'] - HH) / conv_param['stride'])
    W_out = (int)(1 + (Width + 2 * conv_param['pad'] - WW) / conv_param['stride'])
    out = np.zeros([N, F, H_out, W_out])
    n_pad = ((0,0), (0,0), (conv_param['pad'], conv_param['pad']), (conv_param['pad'],conv_param['pad']))
    # print(n_pad)
    x_pad = np.pad(x, pad_width = n_pad, mode = 'constant', constant_values = 0)
    # print(x_pad)
    for idx_n in range(N):
      for idx_f in range(F):
        for idx_h in range(H_out):
          for idx_w in range(W_out):
            x_now = x_pad[idx_n, :, (idx_h*conv_param['stride']):((idx_h*conv_param['stride'])+HH), (idx_w*conv_param['stride']):((idx_w*conv_param['stride'])+WW)]
            out[idx_n, idx_f, idx_h, idx_w] = np.sum(x_now * w[idx_f]) + b[idx_f]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

###x
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache

    N, Chan, Height, Width = x.shape
    F, Chan, HH, WW = w.shape

    x_pad = np.zeros([N, Chan, Height + 2 * conv_param['pad'], Width + 2 * conv_param['pad']])

    for idx_n in range(N):
        for idx_c in range(Chan):
            x_pad[idx_n,idx_c,:,:] = np.pad(x[idx_n,idx_c,:,:], ((conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])), 'constant')

    Hp = int(1 + (Height + 2 * conv_param['pad'] - HH) / conv_param['stride'])
    Wp = int(1 + (Width + 2 * conv_param['pad'] - WW) / conv_param['stride'])

    dx_pad = np.zeros_like(x_pad)
    for idx_n in range(N):
        for idx_c in range(Chan):
            for idx_hp in range(Hp):
                for idx_wp in range(Wp):
                    for idx_f in range(F):
                    	# print(idx_hp)
          				# print(idx_wp)
          				#print(idx_F)
                        dx_pad[idx_n, idx_c, idx_hp * conv_param['stride']:idx_hp * conv_param['stride'] + HH,idx_wp * conv_param['stride']:idx_wp * conv_param['stride'] + WW] \
                         += dout[idx_n, idx_f, idx_hp, idx_wp] *  w[idx_f, idx_c, :, :]

    dx = dx_pad[:,:,conv_param['pad']:-conv_param['pad'],conv_param['pad']:-conv_param['pad']]

    dw = np.zeros([F, Chan, HH, WW])
    for idx_n in range(N):
        for idx_f in range(F):
            for idx_hp in range(Hp):
                for idx_wp in range(Wp):
                    dw[idx_f, :, :, :] += dout[idx_n,idx_f,idx_hp, idx_wp] * x_pad[idx_n,:,idx_hp*conv_param['stride']:idx_hp*conv_param['stride']+HH,idx_wp*conv_param['stride']:idx_wp*conv_param['stride']+WW]



    dout = np.sum(dout, axis=0)
    dout = np.sum(dout, axis=1)
    db = np.sum(dout, axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

###x
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    (N, Chan, Height, Width) = x.shape
    
    W_out = (int)((Width - pool_param['pool_width']) / pool_param['stride'] + 1)
    H_out = (int)((Height - pool_param['pool_height']) / pool_param['stride'] + 1)
    out = np.zeros((N, Chan, H_out, W_out))
    
    for idx_n in range(N):
      for idx_c in range(Chan):
        for idx_h in range(H_out):
          for idx_w in range(W_out):
          	# print(idx_h)
          	# print(idx_w)
            out[idx_n, idx_c, idx_h, idx_w] = np.max(x[idx_n, idx_c, (idx_h*pool_param['stride']):((idx_h*pool_param['stride'])+pool_param['pool_height']), (idx_w*pool_param['stride']):((idx_w*pool_param['stride'])+pool_param['pool_width'])])
            


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

###x
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, Chan, Height, Width = x.shape

    Hp = int(1 + (Height - pool_param['pool_height']) / pool_param['stride'])
    Wp = int(1 + (Width - pool_param['pool_width']) / pool_param['stride'])

    dx = np.zeros_like(x)
    for idx_n in range(N):
        for idx_c in range(Chan):
            for idx_hp in range(Hp):
                for idx_wp in range(Wp):
                    mask = np.zeros([pool_param['pool_height'], pool_param['pool_width']])
                    # print(mask)
                    local_x = x[idx_n, idx_c, idx_hp * pool_param['stride']:idx_hp * pool_param['stride'] + pool_param['pool_height'],idx_wp * pool_param['stride']:idx_wp * pool_param['stride'] + pool_param['pool_width']]
                    mask[np.where(local_x == np.max(local_x))] = 1
                    # print(mask)
                    # print(local_x)
                    dx[idx_n, idx_c, idx_hp * pool_param['stride']:idx_hp * pool_param['stride'] + pool_param['pool_height'],idx_wp * pool_param['stride']:idx_wp * pool_param['stride'] + pool_param['pool_width']] += \
                        mask * dout[idx_n, idx_c, idx_hp, idx_wp]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

    
###x
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
