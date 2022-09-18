import numpy as np
import sys
import matplotlib.pyplot as plt

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        print("Starting BGD...")
        n_samples, n_features = X.shape
        # print(n_samples, n_features)
        loss_list = []
		### YOUR CODE HERE
        self.assign_weights(np.zeros([n_features]))

        epoch = 1
        while epoch <= self.max_iter:
            # print("epoch: ", epoch)
            bgd_grad = np.zeros([n_features])
            bgd_loss = 0
            for i in range(n_samples):
                # print(X.shape, y.shape, X[i].shape, X[i:].shape, y[i:].shape)
                bgd_grad += self._gradient(X[i], y[i])
                bgd_loss += self._loss(X[i], y[i])

            self.W = self.W - self.learning_rate * bgd_grad / n_samples
            # change_W = np.subtract(self.W, prev_W)
            # print("Change in weights: ", change_W)
            loss_list.append(bgd_loss/n_samples)
            epoch += 1

        ### This part plots the Loss Curve
        # plt.plot(list(range(1,len(loss_list)+1)), loss_list)
        # plt.title("BGD Loss Curve")
        # plt.xlabel("Epoch")
        # plt.ylabel("Error")
        # # plt.show()
        # # plt.savefig('./plots/BGD Training Loss.png')
        # plt.clf()
		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        print("Starting mini BGD...")
        n_samples, n_features = X.shape
        # print(n_samples, n_features)
        self.assign_weights(np.zeros([n_features]))
        loss_list = []
        
        for j in range(self.max_iter):
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
                
                mini_bgd_loss = 0
                mini_bgd_grad = np.zeros([n_features])
                for i in range(count * mini_batch_size, count * mini_batch_size + mini_batch_size):
                    mini_bgd_grad += self._gradient(X[i], y[i])
                    mini_bgd_loss += self._loss(X[i], y[i])

                gradient = mini_bgd_grad/mini_batch_size
                self.W = self.W - self.learning_rate * (mini_bgd_grad/mini_batch_size)
                ### To check w1 - w2 = w and the relation between learning rate and gradient/weights
                print("Weights after {} mini-batch: {}".format(count, self.W))
                print("Gradient/Weight: ", np.divide(gradient, self.W))
                mini_bgd_loss = mini_bgd_loss/mini_batch_size
                loss_per_epoch.append(mini_bgd_loss)
                count += 1

            loss_list.append(np.average(loss_per_epoch))
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
        
        ### This part plots the Loss Curve
        # plt.plot(list(range(1,len(loss_list)+1)), loss_list)
        # plt.title("miniBGD Loss Curve")
        # plt.xlabel("Epoch")
        # plt.ylabel("Error")
        # plt.show()
        # plt.savefig('./plots/miniBGD Training Loss.png')
        # plt.clf()
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        print("Starting SGD...")
        n_samples, n_features = X.shape
        # print(n_samples, n_features)
        self.assign_weights(np.zeros([n_features]))

        loss_list = []

        for j in range(self.max_iter):
            sgd_loss = np.zeros([n_features])
            for i in range(n_samples):
                self.W = self.W - self.learning_rate * self._gradient(X[i], y[i])
                sgd_loss += self._loss(X[i], y[i])
            
            loss_list.append(sgd_loss/n_samples)

            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]
        
        ### This part plots the Loss Curve

        # plt.plot(list(range(1,len(loss_list)+1)), loss_list)
        # plt.title("SGD Loss Curve")
        # plt.xlabel("Epoch")
        # plt.ylabel("Error")
        # plt.show()
        # # plt.savefig('./plots/SGD Training Loss.png')
        # plt.clf()
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        log_reg_gradient = ((-1) * (_y) * _x)/(1 + np.exp(_y * np.matmul(np.transpose(self.W), _x)))
        return log_reg_gradient
		### END YOUR CODE
    
    def _loss(self, _x, _y):
        loss = np.log(1+np.exp(-1* _y* np.matmul(np.transpose(self.W), _x)))
        return loss

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        sigmoid_pred_pos = 1 / (1 + np.exp(-1 * np.matmul(X, self.W)))
        sigmoid_pred_neg = 1 - sigmoid_pred_pos
        
        return np.column_stack((sigmoid_pred_pos, sigmoid_pred_neg))
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        sigmoid_preds = 1 / (1 + np.exp(-1*np.matmul(X, self.W)))
        preds = [-1 if p < 0.5 else 1 for p in sigmoid_preds]

        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        n_samples = len(preds)

        return np.sum(preds == y)/n_samples
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

