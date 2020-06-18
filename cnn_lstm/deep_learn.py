from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


"""
a deep learn tool, it contains rnn and cnn now
one hot code is necessary
"""
batch_size = 32

def get_rnn_0(x_train, y_train, x_test, y_test):
    model = Sequential()

    model.add(SimpleRNN(units=16, input_shape=(len(x_train[0]), len(x_train[0][0]))))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(units=2, activation='sigmoid'))

    model.summary()

    #####训练模型
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=30, verbose=2)
    res = model.evaluate(x_test, y_test)

    return res, model.metrics_names


def get_cnn(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(len(x_train[0]), len(x_train[0][0])),
                     data_format="channels_last"))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',  # 'rmsprop'
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=30)
    res = model.evaluate(x_test, y_test)
    return res, model.metrics_names


def get_lstm(x_train, y_train, x_test, y_test):
    model = Sequential()

    model.add(LSTM(units=16, input_shape=(len(x_train[0]), len(x_train[0][0]))))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(units=2, activation='sigmoid'))

    model.summary()

    #####训练模型
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=30, verbose=2)
    res = model.evaluate(x_test, y_test)

    return res, model.metrics_names


def start(_x, _y):
    """
    main function in this function
    :param _x:
    :param _y:
    :return:
    """
    print(_x)
    train_X, test_X, train_y, test_y = train_test_split(_x,
                                                        _y,
                                                        test_size=0.2,
                                                        random_state=1)

    train_X = np.array(train_X)
    test_X = np.array(test_X)
    print(train_X)
    print(train_X.shape)

    # 归一化
    # scaling = MinMaxScaler(feature_range=(0, 1)).fit(train_X)
    # train_X = scaling.transform(train_X)
    # test_X = scaling.transform(test_X)
    # print(np.max(test_X))

    # train_X = np.array(train_X[:, np.newaxis])
    # test_X = np.array(test_X[:, np.newaxis])
    test_y = to_categorical(test_y)
    train_y = to_categorical(train_y)

    predected_rnn, title = get_rnn_0(train_X.copy(), train_y.copy(), test_X.copy(), test_y.copy())

    predected_cnn, title = get_cnn(train_X, train_y, test_X, test_y)

    predected_lstm, title = get_lstm(train_X, train_y, test_X, test_y)

    print("rnn:")
    print(title)
    print(predected_rnn)
    print()
    print("cnn:")
    print(title)
    print(predected_cnn)
    print()
    print("lstm:")
    print(title)
    print(predected_lstm)
