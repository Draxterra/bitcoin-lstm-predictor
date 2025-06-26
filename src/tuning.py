import optuna
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def build_model(trial, input_shape):
    model = Sequential()

    # LSTM Layer 1
    model.add(Bidirectional(LSTM(
        units=trial.suggest_int("units_1", 64, 256, step=32),
        return_sequences=True,
        activation=trial.suggest_categorical("activation_1", ["relu", "tanh"])
    ), input_shape=input_shape))

    # Dropout 1
    model.add(Dropout(trial.suggest_float("dropout_1", 0.1, 0.5)))

    # LSTM Layer 2
    model.add(LSTM(
        units=trial.suggest_int("units_2", 32, 128, step=16),
        return_sequences=False,
        activation=trial.suggest_categorical("activation_2", ["relu", "tanh"])
    ))

    # Dropout 2
    model.add(Dropout(trial.suggest_float("dropout_2", 0.1, 0.5)))

    # Output layer
    model.add(Dense(1))

    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "nadam"])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Nadam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    return model


def objective(trial, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    input_shape = (X.shape[1], X.shape[2])

    model = build_model(trial, input_shape)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        callbacks=[early_stopping],
        verbose=0
    )

    val_loss = min(history.history["val_loss"])
    return val_loss


def tune_lstm(X, y, n_trials=5):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    print("Best params:", study.best_params)
    return study.best_params
