from numpy import array, argmax
from numpy import asarray, float32
from hyperopt.hp import choice, uniform
from hyperopt import Trials, tpe, fmin, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras.activations import relu, elu, selu, softmax
from tensorflow.keras.callbacks import EarlyStopping
from config_reader import min_epochs, max_epochs, max_evals, threshold



def neural_network(training_data: list, number_of_observations: int, number_of_actions: int):
    X = array([i[0] for i in training_data]).reshape(-1, number_of_observations)
    y = array([i[1] for i in training_data])

    nn_space = {
        "units1": choice("units1", range(3, 9)),
        "units2": choice("units2", range(3, 9)),
        "units3": choice("units3", range(3, 9)),
        "units4": choice("units4", range(3, 9)),
        "units5": choice("units5", range(3, 9)),
        "units6": choice("units6", range(3, 9)),
        "units7": choice("units7", range(3, 9)),
        "units8": choice("units8", range(3, 9)),
        "units9": choice("units9", range(3, 9)),
        "units10": choice("units10", range(3, 9)),

        "dropout": uniform("dropout1", 0.2, .8),

        "activation1": choice("activation1", (elu, selu, relu)),
        "activation2": choice("activation2", (elu, selu, relu)),
        "activation3": choice("activation3", (elu, selu, relu)),
        "activation4": choice("activation4", (elu, selu, relu)),
        "activation5": choice("activation5", (elu, selu, relu)),
        "activation6": choice("activation6", (elu, selu, relu)),
        "activation7": choice("activation7", (elu, selu, relu)),
        "activation8": choice("activation8", (elu, selu, relu)),
        "activation9": choice("activation9", (elu, selu, relu)),
        "activation10": choice("activation10", (elu, selu, relu)),
        "activation_output": softmax,

        "lr": choice("lr", range(5, 15)),
        "epochs": choice("epochs", range(min_epochs, max_epochs)),
        "optimizer": choice("optimizer", (Adam, Adadelta, RMSprop)),
        "layer_num": choice("layer_num", (range(5, 11)))
    }
    callback = EarlyStopping(monitor='loss', patience=25)

    def model_creation(nn_space):
        model = Sequential()
        print("layers: {}".format(nn_space["layer_num"]))
        for layer_num in range(nn_space["layer_num"]):
            units_name = "units" + str(layer_num + 1)
            activation_name = "activation" + str(layer_num + 1)
            model.add(Dense(units=2**nn_space[units_name], activation=nn_space[activation_name]))
        model.add(Dropout(rate=nn_space["dropout"]))
        model.add(Dense(number_of_actions, activation=nn_space["activation_output"]))
        return model

    def nn_model(nn_space):
        lr = 1 / 2 ** nn_space["lr"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=42)

        X_train = asarray(X_train.astype(float32))
        X_val = asarray(X_val.astype(float32))

        y_train = asarray(y_train).astype(float32)
        y_val = asarray(y_val).astype(float32)

        model = model_creation(nn_space=nn_space)

        model.compile(
            optimizer=nn_space["optimizer"](lr=lr),
            loss="categorical_crossentropy",
            metrics=["acc"]
        )

        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=nn_space["epochs"],
            verbose=0,
            callbacks=[callback],
            validation_data=(X_val, y_val)
        )

        y_pred = model.predict(X_val)
        y_pred = argmax(y_pred, axis=1)
        y_val = argmax(y_val, axis=1)
        error = f1_score(y_true=y_val, y_pred=y_pred)
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)

        return {
            "type": "regression",
            "loss": error,
            "accuracy": accuracy,
            'model': model,
            "status": STATUS_OK
        }

    def hyperopt_early_stop(trial: Trials, early_stop_args):
        done = False
        is_accurate = False

        for step, trial in enumerate(trial.trials):
            if trial["result"]["loss"] <= threshold:
                done = True
            if trial["result"]["accuracy"] > .85:
                is_accurate = True

        if done and is_accurate:
            stop = True
        else:
            stop = False

        return stop

    trials = Trials()
    fmin(
        trials=trials,
        fn=nn_model,
        space=nn_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        early_stop_fn=no_progress_loss
    )

    best_trial_score = trials.best_trial["result"]["loss"]
    for trial in trials.trials:
        current_score_scaled = trial["result"]["loss"]
        if best_trial_score == current_score_scaled:
            best_model = trial["result"]["model"]
            return best_model
