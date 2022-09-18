import numpy as np
import sys
import matplotlib.pyplot as plt

"""This script implements a multi-class softmax logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """
        print("Starting Multiclass miniBGD...")
		### YOUR CODE HERE
        self.W = np.zeros([X.shape[1], self.k])
        # self.W = np.random.normal(0, 2.5, size=(X.shape[1], self.k))

        # print("W shape: ", self.W.shape)
        # print("Initial W: ", self.W)
        loss_list = []

        for _ in range(self.max_iter):
            val_array = np.array(labels, dtype="int")
            y = np.zeros([val_array.size, val_array.max() + 1])
            y[np.arange(val_array.size), val_array] = 1
            n_samples, n_features = X.shape
            mini_batch_size = 0
            count = 0
            loss_per_epoch = []
            while n_samples > 0:
                if n_samples - batch_size >= 0:
                    n_samples = n_samples - batch_size
                    mini_batch_size = batch_size
                else:
                    mini_batch_size = n_samples
                    n_samples = 0
                
                total_grad = np.zeros([n_features, self.k])
                total_loss = 0
                for i in range(count * mini_batch_size, count * mini_batch_size + mini_batch_size):
                    total_grad += self._gradient(X[i], y[i])
                    total_loss += self._loss(X[i], y[i])

                gradient = total_grad/mini_batch_size
                self.W = self.W - self.learning_rate * (total_grad/mini_batch_size)
                ### To check w1 - w2 = w and the relation between learning rate and gradient/weights
                print("Weights after {} mini-batch: {}".format(count, self.W))
                print("W2 - W1 after {} mini-batch: {}".format(count, self.W[:, 1] - self.W[:, 0]))
                print("Gradient/Weight: ", np.divide(gradient, self.W))
                loss_per_epoch.append(total_loss/mini_batch_size)
                count += 1
            
            loss_list.append(np.average(loss_per_epoch))
            idx = np.random.permutation(X.shape[0])
            X, labels = X[idx], labels[idx]
        # plt.plot(list(range(1,len(loss_list)+1)), loss_list)
        # plt.title("miniBGD Multiclass Loss Curve")
        # plt.xlabel("Epoch")
        # plt.ylabel("Error")
        # plt.show()
        # plt.savefig('./plots/miniBGD Multiclass Loss Curve.png')
        # plt.clf()
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        y_hat = self.softmax(np.matmul(np.transpose(self.W), _x))
        grad = np.outer(_x.reshape(_x.shape[0], 1), y_hat-_y)
        return grad
		### END YOUR CODE
    
    def _loss(self, _x, _y):
        y_hat = self.softmax(np.matmul(np.transpose(self.W), _x))
        loss = -1*np.log(np.matmul(_y, y_hat))
        return loss
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exps = np.exp(x)
        smax = exps/np.sum(exps)
        # print("Softmax shape: ", smax.shape)
        return smax
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_featuress, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        y_temp = np.matmul(X, self.W)
        y_hat = np.array([self.softmax(y_temp[i]) for i in range(y_temp.shape[0])])
        prediction = np.argmax(y_hat , axis=1)
        return prediction
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        n_samples = len(preds)
        return np.sum(preds == labels)/n_samples
		### END YOUR CODE

