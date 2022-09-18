import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.rcParams.update({'figure.figsize':(6,6), 'figure.dpi':100})
    plt.rcParams["figure.autolayout"] = True
    classes = ["-1","1"]
    print(X.shape, y.shape)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("train_features")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    # plt.show()
    plt.savefig("./plots/train_features.png")
    plt.clf()

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
    plt.rcParams.update({'figure.figsize':(6, 6), 'figure.dpi':100})
    plt.rcParams["figure.autolayout"] = True

    print(X.shape, y.shape)
    b = W[0]
    w1, w2 = W[1], W[2]

    c = -b/w2
    m = -w1/w2
    xmin, xmax = min(X[:, 0]), max(X[:, 0])
    ymin, ymax = min(X[:, 1]), max(X[:, 1])

    xd = np.array([xmin, xmax])
    yd = m*xd + c

    scatter = plt.scatter(X[:,0], X[:,1], c= y)
    plt.plot(xd, yd, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    # plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.title(train_result_sigmoid)
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    # plt.show()
    plt.savefig('./plots/train_result_sigmoid.png')
    plt.clf()

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    plt.rcParams.update({'figure.figsize':(6, 6), 'figure.dpi':100})
    plt.rcParams["figure.autolayout"] = True

    print(X.shape, y.shape, W.shape)

    xmin, xmax = min(X[:, 0]), max(X[:, 0])
    ymin, ymax = min(X[:, 1]), max(X[:, 1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.scatter(X[:,0], X[:,1], c= y)
    
    color = ['red', 'green', 'black']
    
    for i in range(len(W[0])):
        b, w1, w2 = W[0][i], W[1][i], W[2][i]
        c = -b/w2
        m = -w1/w2

        xd = np.array([xmin, xmax])
        yd = m*xd + c
        plt.plot(xd, yd, color=color[i], marker='o', linestyle='dashed', linewidth=2, markersize=12)


    plt.title("train_result_softmax")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    # plt.show()
    plt.savefig('./plots/train_result_softmax.png')
    plt.clf()
	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


# ------------Logistic Regression Sigmoid Case------------

##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    final_model = [float('-inf'), [], 0, 0, 0, '']
    mini_bgd_final_model = [float('-inf'), [], 0, 0, 0, '']
    bgd_final_model = [float('-inf'), [], 0, 0, 0, '']
    sgd_final_model = [float('-inf'), [], 0, 0, 0, '']


    lrs = [0.1, 0.2, 0.4, 0.5, 0.75, 1.0]
    max_iters = [50, 100, 200, 400, 500]
    gd_types = ['BGD', 'miniBGD', 'SGD']

    for gd_type in gd_types:
        if gd_type == 'miniBGD':
            mini_batch_sizes = [16, 32, 64, 128]
            for lr in lrs:
                for max_iter in max_iters:
                    logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)
                    for mini_batch_size in mini_batch_sizes:
                        print('miniBGD', lr, max_iter, mini_batch_size)
                        logisticR_classifier.fit_miniBGD(train_X, train_y, mini_batch_size)
                        params = logisticR_classifier.get_params()
                        training_score = logisticR_classifier.score(train_X, train_y)
                        print("training_score: ", training_score)
                        validation_score = logisticR_classifier.score(valid_X, valid_y)
                        print("validation_score: ", validation_score)
                        if validation_score > mini_bgd_final_model[0]:
                            mini_bgd_final_model = [validation_score, params, lr, max_iter, mini_batch_size, 'miniBGD']
                            print(mini_bgd_final_model)
        elif gd_type == 'BGD':
            for lr in lrs:
                for max_iter in max_iters:
                    logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)
                    print('BGD', lr, max_iter)
                    logisticR_classifier.fit_BGD(train_X, train_y)
                    params = logisticR_classifier.get_params()
                    training_score = logisticR_classifier.score(train_X, train_y)
                    print("training_score: ", training_score)
                    validation_score = logisticR_classifier.score(valid_X, valid_y)
                    print("validation_score: ", validation_score)
                    if validation_score > bgd_final_model[0]:
                        bgd_final_model = [validation_score, params, lr, max_iter, 0, 'BGD']
                        print(bgd_final_model)
        elif gd_type == 'SGD':
            for lr in lrs:
                for max_iter in max_iters:
                    logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)
                    print('SGD', lr, max_iter)
                    logisticR_classifier.fit_SGD(train_X, train_y)
                    params = logisticR_classifier.get_params()
                    training_score = logisticR_classifier.score(train_X, train_y)
                    print("training_score: ", training_score)
                    validation_score = logisticR_classifier.score(valid_X, valid_y)
                    print("validation_score: ", validation_score)
                    if validation_score > sgd_final_model[0]:
                        sgd_final_model = [validation_score, params, lr, max_iter, 0, 'SGD']
                        print(sgd_final_model)

    print("bgd_final_model: ", bgd_final_model)
    print("mini_bgd_final_model: ", mini_bgd_final_model)
    print("sgd_final_model: ", sgd_final_model)

    #Best models for each algorithm
    # bgd_final_model = [0.9735449735449735, [ 0.46873881, 10.41498126, -4.74745088], 1.0, 400, 0, 'BGD']
    # mini_bgd_final_model = [0.9788359788359788, [ 1.70682355 17.09388503 -5.49211733], 0.4, 300, 64, 'miniBGD']
    # sgd_final_model = [0.9788359788359788, [ 9.4833702 , 29.21028412,  1.15452184], 0.1, 200, 0, 'SGD']
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    logisticR_classifier = logistic_regression(learning_rate=0.4, max_iter=300)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 64)
    params = logisticR_classifier.get_params()
    print("Final Weights: ", params)
    print("Training Accuracy: ", logisticR_classifier.score(train_X, train_y))
    print("Validation Accuracy: ", logisticR_classifier.score(valid_X, valid_y))
    visualize_result(train_X[:, 1:3], train_y, params)
    visualize_result(valid_X[:, 1:3], valid_y, params)
    ### YOUR CODE HERE

    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_data_test, labels_test = load_data(os.path.join(data_dir, test_filename))

    # # # # # ##### Preprocess raw data to extract features
    test_X_all = prepare_X(raw_data_test)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    test_y_all, test_idx = prepare_y(labels_test)
    ####### For binary case, only use data from '1' and '2'  
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    test_y[np.where(test_y==2)] = -1
    test_data_shape= test_y.shape[0]
    # print("test_data_shape: ", test_data_shape)
    logisticR_classifier = logistic_regression(learning_rate=0.4, max_iter=300)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 64)
    best_params = logisticR_classifier.get_params()

    print("Final Weights: ", best_params )
    print("Training Accuracy: ", score(train_X, train_y))
    print("Validation Accuracy: ", score(valid_X, valid_y))
    print("Test Accuracy: ", score(test_X, test_y))

    visualize_result(test_X[:, 1:3], test_y, best_params)
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all
    visualize_features(train_X[:, 1:3], train_y)

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    final_model = [float('-inf'), [], 0, 0, 0, '']

    lrs = [0.1, 0.2, 0.4, 0.5, 0.75, 1.0]
    max_iters = [50, 100, 200, 400, 500]
    mini_batch_sizes = [16, 32, 64, 128]

    for lr in lrs:
        for max_iter in max_iters:
            logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=lr, max_iter=max_iter,  k= 3)
            for mini_batch_size in mini_batch_sizes:
                print('miniBGD', lr, max_iter, mini_batch_size)
                logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, mini_batch_size)
                params = logisticR_classifier_multiclass.get_params()
                training_score = logisticR_classifier_multiclass.score(train_X, train_y)
                print("training_score: ", training_score)
                validation_score = logisticR_classifier_multiclass.score(valid_X, valid_y)
                print("validation_score: ", validation_score)
                if validation_score > final_model[0]:
                    final_model = [validation_score, params, lr, max_iter, mini_batch_size, 'miniBGD']
                    print(final_model)
    print("Final Model: ", final_model)
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=200,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 128)
    params = logisticR_classifier_multiclass.get_params()
    print("Params: ", params)
    training_score = logisticR_classifier_multiclass.score(train_X, train_y)
    print("training_score: ", training_score)
    validation_score = logisticR_classifier_multiclass.score(valid_X, valid_y)
    print("validation_score: ", validation_score)

    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    raw_data_test, labels_test = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_data_test)
    test_y, idx = prepare_y(labels_test)

    testing_score = logisticR_classifier_multiclass.score(test_X, test_y)
    print("testing_score: ", testing_score)
    visualize_result_multi(test_X[:, 1:3], test_y, params)
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=10000,  k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 32)
    lrm_convergence_params = logisticR_classifier_multiclass.get_params()
    print("LRM Convergence Params: ", lrm_convergence_params)
    training_score = logisticR_classifier_multiclass.score(train_X, train_y)
    print("training_score: ", training_score)
    validation_score = logisticR_classifier_multiclass.score(valid_X, valid_y)
    print("validation_score: ", validation_score)
    ### END YOUR CODE


    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.4, max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 64)
    convergence_params = logisticR_classifier.get_params()
    print("LR convergence_params: ", convergence_params)
    training_score = logisticR_classifier.score(train_X, train_y)
    print("training_score: ", training_score)
    validation_score = logisticR_classifier.score(valid_X, valid_y)
    print("validation_score: ", validation_score)
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0 

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=1,  k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 256)

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    logisticR_classifier = logistic_regression(learning_rate=0.2, max_iter=1)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 256)
    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
