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
    losses = np.zeros(num_samples)

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
                prediction[i] += sample[j] * class_weight_vector[j]

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
                dW[j][i] += (sample[j] * (y_hat[i] - delta_yi))

        # for k in range(num_classes):
        #     # delta_yi: 0 if not correct label, 1 if correct label
        #     delta_yi = 0
        #     if k == y_sample:
        #         delta_yi = 1
        #
        #
        #     for d in range(D):
        #         #if k == y[sample_index]:
        #         dW[d, k] += X.T[d, sample_index] * (y_hat[k] - delta_yi)
        #         #else:
        #             #dW[d, k] += X.T[d, sample_index] * y_hat[k]

    # calculate mean cross entropy loss and mean gradients
    sum_of_losses = 0
    for l_j in losses:
        sum_of_losses += l_j
    loss = sum_of_losses / num_samples
    dW /= num_samples

    # add L2 regularization
    # sum (w_i^2)
    loss += reg * np.sum(W ** 2)
    dW += reg * W

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

    num_classes = W.shape[1]
    num_samples = X.shape[0]
    D = X.shape[1]

    # transform y to one-hot
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1

    y_hat = perform_forward_pass_vecotrized(X, W)

    # compute cross entropy loss using the one hot array to create some kind of mask
    # true label predictions
    # after that y_hat is an array with the same shape but it now only contains predictions for the
    # real class label and is otherwise 0
    #y_hat_m = np.multiply(y_hat, y_one_hot)
    #losses = - np.log(np.multiply(y_one_hot, y_hat_m), where=y_one_hot.astype(bool))
    losses = - np.log(y_hat[range(num_samples), y])
    #true_predictions = y_hat[range(num_samples), y]
    #dW = X.T.dot(y_hat - y_one_hot)
    dW = np.dot(X.T, (y_hat - y_one_hot))
    dW /= num_samples

    loss = np.sum(losses) / num_samples

    # add L2 regularization
    # sum (w_i^2)
    loss += reg * np.sum(W ** 2)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def perform_forward_pass_vecotrized(X, W):
    # forward pass (matrix multiplication)
    prediction = X.dot(W)

    # calculate softmax for prediction
    # take care of potential numerical issues
    prediction -= np.max(prediction, axis=1, keepdims=True)
    return np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)

# from dl4cv.data_utils import load_CIFAR10
#
# # Load the raw CIFAR-10 data
# cifar10_dir = 'C:\\Users\\felix\\OneDrive\\Studium\\Studium\\4. Semester\\DL4CV\Exercises\\01\\dl4cv\\exercise_1\\datasets'
# X, y = load_CIFAR10(cifar10_dir)
#
# # Split the data into train, val, and test sets. In addition we will
# # create a small development set as a subset of the data set;
# # we can use this for development so our code runs faster.
# num_training = 48000
# num_validation = 1000
# num_test = 1000
# num_dev = 500
#
# assert (num_training + num_validation + num_test) == 50000, 'You have not provided a valid data split.'
#
# # Our training set will be the first num_train points from the original
# # training set.
# mask = range(num_training)
# X_train = X[mask]
# y_train = y[mask]
#
# # Our validation set will be num_validation points from the original
# # training set.
# mask = range(num_training, num_training + num_validation)
# X_val = X[mask]
# y_val = y[mask]
#
# # We use a small subset of the training set as our test set.
# mask = range(num_training + num_validation, num_training + num_validation + num_test)
# X_test = X[mask]
# y_test = y[mask]
#
# # We will also make a development set, which is a small subset of
# # the training set. This way the development cycle is faster.
# mask = np.random.choice(num_training, num_dev, replace=False)
# X_dev = X_train[mask]
# y_dev = y_train[mask]
#
# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_val = np.reshape(X_val, (X_val.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))
# X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
#
# # As a sanity check, print out the shapes of the data
# print('Training data shape: ', X_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Test data shape: ', X_test.shape)
# print('dev data shape: ', X_dev.shape)
#
# mean_image = np.mean(X_train, axis=0)
# print(mean_image[:10]) # print a few of the elements
#
# X_train -= mean_image
# X_val -= mean_image
# X_test -= mean_image
# X_dev -= mean_image
#
# X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
# X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
# X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
# X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
#
# print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
#
# # Generate a random softmax weight matrix and use it to compute the loss.
# W = np.random.randn(3073, 10) * 0.0001
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
# loss2, grad2 = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
#
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
#
# # We use numeric gradient checking as a debugging tool.
# # The numeric gradient should be close to the analytic gradient.
# from dl4cv.gradient_check import grad_check_sparse
# f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad, num_checks=3)
#
# # Do another gradient check with regularization
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 1e2)
# f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 1e2)[0]
# grad_numerical = grad_check_sparse(f, W, grad, num_checks=3)
#
# # As a rough sanity check, our loss should be something close to -log(0.1).
# #print('loss: %f' % loss)
# #print('sanity check: %f' % (-np.log(0.1)))

