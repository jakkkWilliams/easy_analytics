import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import statsmodels.api as sm
rng = np.random

def data_regularize(data):
    ss = StandardScaler()
    ss.fit(data)
    return ss.transform(data)

def calc_state(X, W_in, b_in, W_hid, b_hid, activation_function = 'relu'):
    nHidLayers = len(W_hid)
    p_in = tf.add(tf.matmul(W_in, X), b_in)
    if activation_function == 'relu':
        p_in = tf.nn.relu(p_in)
    elif activation_function == 'sigmoid':
        p_in = tf.sigmoid(p_in)
    else:
        raise ValueError('Must specify valid activate function')
    
    for i in range(nHidLayers):
        if i == 0: # First Hidden Layer
            pred = tf.add(tf.matmul(W_hid[i], p_in), b_hid[i])
        else:
            pred = tf.add(tf.matmul(W_hid[i], pred), b_hid[i])
        if activation_function == 'relu':
            pred = tf.nn.relu(pred)
        elif activation_function == 'sigmoid':
            pred = tf.sigmoid(pred)
        else:
            raise ValueError('Must specify valid activate function')
    return pred

def calc_linear_state(X, W, b, activation_function = 'linear'):
    linear = tf.add(tf.matmul(W, X), b)
    if activation_function == 'linear':
        pred = linear
    elif activation_function == 'poisson':
        pred = tf.exp(linear)
    elif activation_function == 'sigmoid':
        pred = tf.sigmoid(linear)
    else:
        raise ValueError('Must specify valid activate function')
    return pred

def create_network(train_X, train_Y, learning_rate = 0.0001, hidLayers = [5, 5], batchsize = 1, activation_function = 'relu'):
    global nobs, nvar, X, Y, W_in, b_in, W_hid, b_hid, loss, opt
    nobs, nvar = train_X.shape
    nHidLayers = len(hidLayers)
    
    X = tf.placeholder("float", [nvar, None], name = "X")
    Y = tf.placeholder("float", [1, None], name = "Y")

    W_in = tf.Variable(tf.random_normal([hidLayers[0], nvar]), name = "w1")
    b_in = tf.Variable(tf.random_normal([1]), name = 'b1')

    W_hid = []
    b_hid = []
    for i in range(nHidLayers):
        w_name = 'w_h'+str(i+1)
        b_name = 'b_h'+str(i+1)
        if i+1 < nHidLayers:
            nextSize = hidLayers[i+1]
        else:
            nextSize = 1
        w = tf.Variable(tf.random_normal([nextSize, hidLayers[i]]), name = w_name)
        b = tf.Variable(tf.random_normal([1]), name = b_name)
        W_hid.append(w)
        b_hid.append(b)

    pred = calc_state(X, W_in, b_in, W_hid, b_hid, activation_function = activation_function)

    loss = tf.sqrt(tf.reduce_mean(tf.square(pred - Y)))
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


def implement_glm(train_X, train_Y, test_X, test_Y, activation_function = 'linear'):
    train_X = sm.add_constant(train_X)
    test_X = sm.add_constant(test_X)
    if activation_function == 'linear':
        model = sm.GLM(train_Y, train_X)
    elif activation_function == 'poisson':
        model = sm.GLM(train_Y, train_X, family=sm.families.Poisson())
    else:
        raise ValueError('Specify valid function for the GLM.')
    result = model.fit()

    train_pred = result.predict(train_X)
    train_rmse = np.sqrt(np.mean(np.square(train_pred - train_Y)))
    test_pred = result.predict(test_X)
    test_rmse = np.sqrt(np.mean(np.square(test_pred - test_Y)))
    return train_rmse, test_rmse

def prepare_batch(x, batchsize):
    nobs, nvar = x.shape
    nbatch = int(nobs/batchsize)
    return nbatch

def get_next_batch(X, Y, nbatch, batchsize, batch_num):
    begin = batch_num*batchsize
    end = min(X.shape[0], (batch_num+1)*batchsize)
    x = X[begin:end, :]
    y = Y[begin:end, :]
    return x, y
    
def learn_minibatch(sess, opt, loss, train_X, train_Y, nepoch = 1000, display_step = 50, batchsize = 1, result_with_no_step = True):
    for epoch in range(nepoch):
        nbatch = prepare_batch(x = train_X, batchsize = batchsize)
        for i in range(nbatch):
            x, y = get_next_batch(X = train_X, Y = train_Y, nbatch = nbatch, batchsize = batchsize, batch_num = i)
            c = opt.run(feed_dict={X: x.T, Y: y.T})
            perf_learn = sess.run(loss, feed_dict={X: x.T, Y: y.T})
        if not result_with_no_step:
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), 'loss(learn):', perf_learn)
    return perf_learn
    
def learn_fullbatch(sess, opt, loss, train_X, train_Y, nepoch = 1000, display_step = 50, result_with_no_step = True):
    for epoch in range(nepoch):
        c = opt.run(feed_dict={X: train_X.T, Y: train_Y.T})
        perf_learn = sess.run(loss, feed_dict={X: train_X.T, Y: train_Y.T})
        if not result_with_no_step:
            if epoch % display_step == 0:
                #print("Epoch:", '%04d' % (epoch+1), 'loss(learn):', perf_learn)
                print("Epoch:", '%04d' % (epoch+1), 'loss(learn):', perf_learn)
    return perf_learn

def test_with_learnt_network(sess, loss, test_X, test_Y, activation_function = 'relu'):
    nobs_test, nvar_test = test_X.shape
    test_pred = calc_state(X = test_X.T, W_in = W_in, W_hid = W_hid, b_in = b_in, b_hid = b_hid, activation_function = activation_function)
    test_loss = tf.sqrt(tf.reduce_mean(tf.square(test_pred - test_Y.T)))
    test_perf = sess.run(test_loss)
    return test_perf

def create_dataset(d, X, Y, ratio = 0.75, regularize = True):
    x = np.array(d[X], dtype=np.float32)
    y = np.array(d[Y], dtype=np.float32)
    if regularize:
        x = data_regularize(x)
        y = data_regularize(y)
    
    if x.shape[0] == y.shape[0]:
        nobs = x.shape[0]
        ntrain = int(nobs*ratio)
    else:
        raise ValueError('NOBS X and Y differ.')
    
    train_X = x[0:ntrain, :]
    train_Y = y[0:ntrain, :]
    test_X = x[ntrain+1:, :]
    test_Y = y[ntrain+1:, :]
    return train_X, train_Y, test_X, test_Y

def create_result_file(result_filename):
    df_result = pd.DataFrame(colnames = ['model_name', 'target', 'Epoch', 'learning_rate', 'batchsize', 'HidLayers', 'ActivateFunc', 'Perf(Learn)', 'Perf(Test)'])
    df_result.to_csv(result_filename)
    
def write_result(result_filename, model_name, target, nepoch, learning_rate, batchsize, hidLayers, activatione_function, perf_learn, perf_test):
    fname = result_filename + '.csv'
    if not os.path.isfile(fname):
        df_result = pd.DataFrame({
            'model_name': model_name,
            'target': target,
            'Epoch': nepoch,
            'learning_rate': learning_rate, 
            'batchsize': batchsize, 
            'HidLayers': hidLayers, 
            'ActivateFunc': activation_function, 
            'Perf(Learn)': perf_learn, 
            'Perf(Test)': perf_test
        })
    else:
        df_result = pd.read_csv(fname, index_col=0)
        new_record = pd.DataFrame({
            'model_name': model_name,
            'target': target,
            'Epoch': nepoch,
            'learning_rate': learning_rate, 
            'batchsize': batchsize, 
            'HidLayers': hidLayers, 
            'ActivateFunc': activation_function, 
            'Perf(Learn)': perf_learn, 
            'Perf(Test)': perf_test
        })
        df_result = pd.concat([df_result, new_record])
    df_result.to_csv(fname)
    print('Saved into csv: ', fname)

def print_timestamp():
    time = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S")
    print('Timestamp:', time)
    
def run_one_loop(model_name, data, varlist, target, hidden_layers, activation_function, 
    display_step, result_with_no_step, result_filename, 
    learn_ratio = 0.75, learning_rate = 0.0001, nepoch = 1000, batchsize = 1, save_result = True
                ):

    # CREATE TRAINING DATASET
    is_deep = (len(hidden_layers)>0)
    #train_X, train_Y, test_X, test_Y = create_dataset(data, X=varlist, Y=target, ratio=learn_ratio, regularize = is_deep)
    train_X, train_Y, test_X, test_Y = create_dataset(data, X=varlist, Y=target, ratio=learn_ratio, regularize = True)

    # USE TENSORFLOW IF DEEP LEARNING
    if is_deep:
        import tensorflow as tf
        print('[Deep Learning]')
        # VERIFY SETTINGS
        if activation_function in ['linear', 'poisson']:
            raise ValueError('Use linear/relu for activation when Deep Learning.')
        # CONSTRUCT DEEPNET WITH THE SPECIFIED SETTINGS
        create_network(
            train_X = train_X,
            train_Y = train_Y,
            learning_rate = learning_rate,
            batchsize = batchsize,
            hidLayers = hidden_layers,
            activation_function = activation_function
        )
        # START LEARNING PROCESS
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # LEARNING
            perf_learn = learn_minibatch(
                sess,opt, loss,
                train_X = train_X,
                train_Y = train_Y,
                nepoch = nepoch,
                result_with_no_step = result_with_no_step, 
                display_step=display_step,
                batchsize = batchsize
            )
            # TEST
            perf_test = test_with_learnt_network(
                sess, loss,
                test_X = test_X,
                test_Y = test_Y,
                activation_function = activation_function
            )
        if save_result:
            write_result(
                result_filename = result_filename,
                model_name = model_name, target = target,
                nepoch = nepoch, learning_rate = learning_rate, 
                batchsize = batchsize, hidLayers = str(hidden_layers), 
                activate_function = activation_function, 
                perf_learn = perf_learn, perf_test = perf_test
            )
    
    # USE STATSMODELS IF LINEAR REGRESSION
    else:
        import statsmodels.api as sm
        print('[Linear Regression]')
        if activation_function in ['relu']:
            raise ValueError('Use linear/poisson for activation when Linear Regression.')
        # EVALUATE WITH THE SPECIFIED MODEL
        perf_learn, perf_test = implement_glm(
            train_X = train_X,
            train_Y = train_Y,
            test_X = test_X,
            test_Y = test_Y
        )
        if save_result:
            write_result(
                result_filename = result_filename,
                model_name = model_name, target = target,
                nepoch = '-', learning_rate = '-', 
                batchsize = '-', hidLayers = '-', 
                activation_function = activation_function, 
                perf_learn = perf_learn, perf_test = perf_test
            )
    print("[Calc Finished]\n", '- loss(learn):', perf_learn, '\n', '- loss(test):', perf_test)
    #return perf_learn, perf_test

def display_results(result_filename):
    fname = result_filename + '.csv'
    df = pd.read_csv(fname, index_col=0)
    display(df)

