from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras.activations import relu,  elu, selu, softmax
from tensorflow.keras.callbacks import EarlyStopping
from config_reader import LR
from numpy import array


def neural_network(training_data: list, number_of_observations: int):
    X = array([i[0] for i in training_data]).reshape(-1, number_of_observations)
    y = array([i[1] for i in training_data])

    model = Sequential()
    model.add(Dense(units=32, activation=relu))
    model.add(Dense(units=64, activation=relu))
    model.add(Dense(units=128, activation=relu))
    model.add(Dense(units=64, activation=relu))
    model.add(Dense(units=32, activation=relu))
    model.add(Dropout(rate=.5))
    model.add(Dense(units=2, activation=softmax))

    model.compile(
        optimizer=Adam(lr=LR),
        loss="categorical_crossentropy")

    model.fit(
        X, y,
        batch_size=32,
        epochs=1000,
        verbose=1)

    return model
