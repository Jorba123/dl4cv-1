import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_classes = W.shape[1]
    num_samples = X.shape[0]
    D = X.shape[1]
    # compute scores based on W and X ("forward" pass)
    f = np.zeros(num_classes)
    losses = np.zeros(num_samples)
    # softmax predictions
    #y_hat = np.zeros((num_samples, num_classes))

    # transform y to the same format (one-hot). Strictly speaking this contains np loops but since it just makes the
    # calculation easier this should be fine
    #y_oh = np.zeros((num_samples, num_classes))
    #y_oh[np.arange(num_samples), y] = 1
    #print('One Hot y: ', y_oh[0])
    #print('y: ', y[0])
    print('Num Classes ' + str(num_classes))

    # do the calculation for each sample in the batch
    for sample_index in range(num_samples):
        sample = X[sample_index]
        prediction = np.zeros(num_classes)
        # iterate over classes and do matrix multiplication with explicit loops
        for i in range(num_classes):
            # get part of weight vector that corresponds to the i-th class. This will be a vector with shape (
            class_weight_vector = W[:, i]
            # iterate over pixels
            for j in range(D):
                prediction[i] = sample[j] * class_weight_vector[j]

            f[i] += prediction[i]

        # calculate softmax to get y_hat (predicted class labels)
        y_hat = calculate_softmax(prediction)
        assert sum(y_hat) - 1.0 < 0.001, 'Sum is not 1.0 {0}'.format(sum(y_hat))
        # calculate cross entropy loss
        # y_sample contains the index of the prediction value we need for the cross entropy loss calculation
        y_sample = y[sample_index]
        y_loss = np.log(y_hat[y_sample]) * (-1)
        losses[sample_index] = y_loss


        # Calculate gradients. Get the Jacobian of the cross entropy loss function w.r.t to the weights factoring in
        # the chain rule.
        # the calculation of the gradients of xent, softmax and weight multiplication can be simplified to
        # D_ij xent (W) = x_j * (S_i - delta_yi)
        # S_i -> Softmax result at i
        # delta_yi Kronecker delta. y = Correct label
        # x_j Input value at j
        # For this we have to iterate again over each pixel value and each class

        for i in range(num_classes):
            # delta_yi: 0 if not correct label, 1 if correct label
            delta_yi = 0
            if i == y_sample:
                delta_yi = 1

            # iterate over pixel data
            for j in range(D):
                dW[j][i] += sample[j] * (prediction[i] - delta_yi)

    # calculate mean cross entropy loss and mean gradients
    sum_of_losses = 0
    for l_j in losses:
        sum_of_losses += l_j
    loss = sum_of_losses / num_samples
    dW /= num_samples



    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def calculate_softmax(f):
    num_classes = f.shape[0]

    # calculate softmax and cross entropy loss for the softmax
    # to prevent numeric instability due to large numbers from exponents, normalize so that max is 0
    max_score = np.max(f)
    for i in range(num_classes):
        f[i] -= max_score

    # calculate divisor (sum(exp(f))
    sum_of_scores = 0
    for f_i in f:
        sum_of_scores += np.exp(f_i)

    softmax = np.zeros(num_classes)

    for j in range(num_classes):
        softmax[j] = np.exp(f[j]) / sum_of_scores

    return softmax

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

