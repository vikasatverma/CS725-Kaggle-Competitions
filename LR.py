import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)

class Scaler():
    def __init__(self):
        pass
    def __call__(self,features, is_train=False):
        Xnorm = features.copy()
        minx = np.min(X[:,1:])
        maxx = np.max(X[:,1:])
        Xnorm[:,1:] = (X[:,1:]-minx)/(maxx-minx)
        return Xnorm

def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''
    
    
    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''
    df = pd.read_csv(csv_path)
    if(is_train):
        
        Q1 = df[' shares'].quantile(0.25)
        Q3 = df[' shares'].quantile(0.75)
        IQR = Q3 - Q1
        LTV= Q1 - (1.5 * IQR)
        UTV= Q3 + (1.5 * IQR)
        df = df.drop(df[df[' shares'] > UTV].index)
        df.shape
        df.drop(columns=[' shares'],inplace=True)
        
    df = df.assign(bias=1) 
    ds = df.to_numpy()
    
    return (ds)
    
    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''


def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    df = pd.read_csv(csv_path)
    Q1 = df[' shares'].quantile(0.25)
    Q3 = df[' shares'].quantile(0.75)
    IQR = Q3 - Q1
    LTV= Q1 - (1.5 * IQR)
    UTV= Q3 + (1.5 * IQR)
    df = df.drop(df[df[' shares'] > UTV].index)
    df.shape
    
    ds = df.to_numpy()
    ds = ds[:,-1:]
    return (ds)
    
    
def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''
    trans_feature_matrix = feature_matrix.transpose()
    result = np.matmul(trans_feature_matrix,feature_matrix)
    result = np.add(result, C*np.eye(feature_matrix.shape[1]))
    result = np.linalg.inv(result)
    result = np.matmul(result, trans_feature_matrix)
    result = np.matmul(result, targets)
    
    df = pd.DataFrame(result)
    df.to_csv('weights.csv',index=False)
    
    return result
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape m x 1
    '''
    

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''
    result = np.matmul(feature_matrix, weights)

#     df = pd.DataFrame(result)
#     df.to_csv('pred.csv',index=False)
    return result
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    ret =  np.square(np.subtract(get_predictions(feature_matrix, weights), targets)).mean()
    
    return ret
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    return np.sum(np.square(weights))
    '''
    Arguments
    weights: numpy array of shape n x 1
    '''

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''
    return mse_loss(feature_matrix, weights, targets) + C * l2_regularizer(weights)
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    
    return ((2/weights.shape[0])*(feature_matrix.T.dot((feature_matrix.dot(weights))-targets)))

    

    
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''
    idx = np.random.choice(np.arange(len(targets)), batch_size, replace=False)
    x_sample = feature_matrix[idx]
    y_sample = targets[idx]
    return (x_sample,y_sample)

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    raise NotImplementedError
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''
    
    return np.ones(n).reshape(n,1)
    '''
    Arguments
    n: int
    '''
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''
    return weights - lr*gradients
    
    
    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    raise NotImplementedError

def early_stopping(thetanew, theta0, TOL=1e-20):
    if np.sum((thetanew - theta0)**2) < TOL:
            print('Convergence achieved.')
            return True
    return False
    raise NotImplementedError

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    n = train_feature_matrix.shape[1]
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        gradients = compute_gradients(features, weights, targets, C)
        
        
        
        #update weights
        weights = update_weights(weights, gradients, lr)
        
        if step%eval_steps == 0:
            
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss,))

        '''
        implement early stopping etc. to improve performance.
        '''
    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
#     print loss
    return loss







def is_close(v1,v2, allowed_error=0.0001):
    if np.sum((v1-v2)**2) < allowed_error:
        return True
    else:
        return False


def test_case_1():
    features = np.eye(3)
    targets = np.array([5.0,13.0,2.0]).reshape(3,1)

    weights = analytical_solution(feature_matrix=features, targets=targets, C=0.0)

    if np.all(weights==targets):
        print('\nyour analytical solution passed this test case!\n')
    else:
        print('\noops! your analytical solution is failed this test case!\n')

    gradients_1 = compute_gradients(feature_matrix=features, weights=weights, targets=targets, C=0.0)
    #print(gradients_1)
    check_gradients_1 = np.all(gradients_1==0)

    gradients_2 = compute_gradients(feature_matrix=features, weights=np.zeros([3,1]), targets=targets, C=0.0)
    true_gradients_2 = -2/3 * targets
    check_gradients_2 = is_close(true_gradients_2,gradients_2)

    if check_gradients_1 and check_gradients_2:
        print('\nyour compute_gradients solution passed this test case!\n')
    else:
        print('\noops! your compute_gradients solution is failed this test case!\n')

    updated_weights = update_weights(weights=np.zeros([3,1]), gradients=gradients_2, lr=1)
    check_updated_weights = is_close(-updated_weights, gradients_2)

    if check_updated_weights:
        print('\nyour update_weights solution passed this test case!\n')
    else:
        print('\noops! your update_weights solution is failed this test case!\n')






if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    test_case_1()
    training_data_csv="../input/programming-assignment-1/train.csv"
    dev_data_csv="../input/programming-assignment-1/dev.csv"
    test_data_csv = "../input/programming-assignment-1/test.csv"
    
    train_features, train_targets = get_features(training_data_csv,True,scaler), get_targets(training_data_csv)
    dev_features, dev_targets = get_features(dev_data_csv,True,scaler), get_targets(dev_data_csv)
    test_features = get_features(test_data_csv,False,scaler)
    
    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    
    print(get_predictions(test_features, a_solution))
    df = pd.DataFrame(get_predictions(test_features, a_solution))
    df.to_csv('pred.csv',index=True)
    print(train_loss)

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=1e-15,
                        C=1e-8,
                        batch_size=32,
                        max_steps=4000000,
                        eval_steps=50)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    
    
    
    test_features = get_features('../input/programming-assignment-1/test.csv',False,scaler)
    predictions = np.around(get_predictions(test_features, gradient_descent_soln))
    df = pd.DataFrame(predictions)
    df.to_csv('predicted.csv',index=False)
    print("Done")


