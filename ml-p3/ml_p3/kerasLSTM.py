import numpy as np
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from ml_p3.utils.utils import prepare_eth_data


def prepare_data_for_keras(Y_train, Y_test):
    # Keras expects targets as (samples, features)
    # Currently Y_train shape is (num_samples, horizon, 1) => (num_samples, 1, 1)
    # We can reshape this to (num_samples, 1)
    Y_train = Y_train.squeeze(axis=1)
    Y_test = Y_test.squeeze(axis=1)
    return Y_train, Y_test


def build_model(lookback, input_dim, hidden_dim, output_dim, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(lookback, input_dim)))
    model.add(Dense(output_dim))
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    model.summary()
    return model


def train_model(model, X_train, Y_train, epochs=50, batch_size=32):
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    return history


def evaluate_model(model, X, Y, data_type="Training"):
    Y_pred = model.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, Y_pred)
    scores = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    # print(f"{data_type} Scores:")
    # print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")
    return Y_pred, scores


def plot_predictions(Y_true, Y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true, label="Actual")
    plt.plot(Y_pred, label="Predicted")
    plt.title("Keras LSTM Predictions")
    plt.xlabel("Time")
    plt.ylabel("Scaled Price")
    plt.legend()
    plt.show()


def main():
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(script_dir, "data", "resampled_eth_data.csv")
    data_file_path = os.path.normpath(data_file_path)
    lookback = 1
    future_steps = 1
    hidden_dim = 16
    output_dim = 1
    epochs = 50
    batch_size = 32

    X_train, Y_train, X_test, Y_test, _, _ = prepare_eth_data(
        data_file_path, lookback=lookback, horizon=future_steps
    )

    Y_train, Y_test = prepare_data_for_keras(Y_train, Y_test)

    input_dim = X_train.shape[-1]

    model = build_model(lookback, input_dim, hidden_dim, output_dim)

    history = train_model(model, X_train, Y_train, epochs, batch_size)

    Y_train_pred = evaluate_model(model, X_train, Y_train, data_type="Training")

    Y_test_pred, _ = evaluate_model(model, X_test, Y_test, data_type="Testing")

    plot_predictions(Y_test, Y_test_pred)
    model.save("keras_lstm.keras")


if __name__ == "__main__":
    main()
