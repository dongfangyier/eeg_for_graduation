# cnn model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import to_categorical
from matplotlib import pyplot
import os



'''
一个cnn的模型，未完成
'''

# load a single file as a numpy array
def load_file(filepath):
    print("Start load_file")
    dataframe = read_csv(filepath, header=None)
    print("Finish load_file")
    return dataframe.values


data_max = -1
data_min = -1


# 用于三个维度的归一化
def normalization(data):
    global data_min
    global data_max
    if data_max == -1:
        data_min = np.min(data)
        data_max = np.max(data)
    _range = data_max - data_min
    return (data - data_min) / _range


# load a list of files and return as a 3d numpy array
def load_group(filenames, filepath):
    print("Start load_group")
    loaded = list()
    for name in filenames:
        data = load_file(filepath + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    print("Finish load_group")
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(filepath):
    print("Start load_dataset_group")
    # load all 9 files as a single array
    filenames = os.listdir(filepath)
    del filenames[filenames.index('y.csv')]
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(filepath + 'y.csv')
    print("Finish load_dataset_group")
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(filepath):
    print("Start load_dataset")
    # load all train
    trainX, trainy = load_dataset_group(filepath + 'train/')

    testX, testy = load_dataset_group(filepath + 'test/')

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX, trainy, testX, testy


# fit and evaluate a cnn model
def evaluate_model(trainX, trainy, testX, testy,i):
    print("Start evaluate_model")
    verbose, epochs, batch_size = 1, 30, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()

    # 后续在输入格式
    model = Sequential()
    model.add(Convolution1D(nb_filter=50, kernel_size=(1,5),activation='linear',filter_length=1, input_shape=(n_timesteps, n_features)))
    model.add(Convolution1D(nb_filter=20, kernel_size=(1,3),activation='linear',filter_length=1, input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=(2, 1)))
    model.add(Activation('relu'))
    model.add(Dense(96))
    model.add(Dense(96))
    model.add(Activation('sigmod'))

    model.add(Dense(3))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
    print()
    print(i)
    model.save('model/model_'+str(i)+'.h5')
    print("Finish evaluate_model")
    return accuracy


# summarize scores
def summarize_results(scores):
    print("Start summarize_results")
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    print("Finish summarize_results")


# run an experiment
def run_experiment(filepath, repeats=8):
    # load data
    print("Start run_experiment")
    trainX, trainy, testX, testy = load_dataset(filepath)
    # repeat experiment
    scores = list()
    i=0
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy,i)
        i+=1
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    print("Finish run_experiment")


# run the experiment
filepath = '/home/rbai/eeg_lstm_0706/data/'
run_experiment(filepath)
