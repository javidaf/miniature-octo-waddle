import os
import pickle
from matplotlib import pyplot as plt
from ml_p3.LSTM import LSTMNetwork
from ml_p3.utils.utils import load_lstm_model
from keras.api.models import load_model
from ml_p3.kerasLSTM import evaluate_model, prepare_data_for_keras, prepare_eth_data
from ml_p3.NN_classification import NN_classification


def get_paths():
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        script_dir, "..", "tests", "data", "resampled_eth_data.csv"
    )
    lstm_model_path = os.path.normpath(
        os.path.join(script_dir, "models", "custom_lstm.pkl")
    )
    keras_lstm_model_path = os.path.normpath(
        os.path.join(script_dir, "models", "keras_lstm.keras")
    )
    return data_path, lstm_model_path, keras_lstm_model_path


def load_models(lstm_model_path, keras_lstm_model_path):
    with open(lstm_model_path, "rb") as f:
        model_state = pickle.load(f)
    custom_model = load_lstm_model(lstm_model_path, LSTMNetwork)
    keras_model = load_model(keras_lstm_model_path)
    future_steps = model_state["future_steps"]
    return custom_model, keras_model, future_steps


def prepare_datasets(data_path, lookback, horizon):
    _, Y_train, X_test, Y_test, _, _ = prepare_eth_data(
        data_path, lookback=lookback, horizon=horizon
    )
    _, Y_test_keras = prepare_data_for_keras(Y_train, Y_test)
    return X_test, Y_test, Y_test_keras


def evaluate_models(custom_model, keras_model, X_test, Y_test, Y_test_keras):
    y_pred_custom = custom_model.predict(X_test)
    y_test_scores_custom = custom_model.scores(X_test, Y_test)
    y_pred_keras, y_test_scores_keras = evaluate_model(
        keras_model, X_test, Y_test_keras
    )
    return y_pred_custom, y_test_scores_custom, y_pred_keras, y_test_scores_keras


def plot_results(Y_test, y_pred_custom, y_pred_keras):
    plt.plot(Y_test.flatten(), label="Actual")
    plt.plot(y_pred_custom.flatten(), label="Custom LSTM Predicted")
    plt.plot(
        y_pred_keras.flatten(),
        label="Keras LSTM Predicted(Not same as in report!!, Forgot to save model)",
    )
    plt.legend()
    plt.show()


def main():

    print("Running cell classification")
    NN_classification()

    print("Running LSTM")

    optimal_lookback_custom = 16
    print("Hello from ml_p3")
    data_path, lstm_model_path, keras_lstm_model_path = get_paths()
    custom_model, keras_model, future_steps = load_models(
        lstm_model_path, keras_lstm_model_path
    )
    X_test, Y_test, Y_test_keras = prepare_datasets(
        data_path, optimal_lookback_custom, future_steps
    )
    y_pred_custom, y_test_scores_custom, y_pred_keras, y_test_scores_keras = (
        evaluate_models(custom_model, keras_model, X_test, Y_test, Y_test_keras)
    )
    print("Testing scores custom:", y_test_scores_custom["r2"])
    print("Testing scores keras:", y_test_scores_keras["r2"])
    plot_results(Y_test, y_pred_custom, y_pred_keras)


if __name__ == "__main__":
    main()
