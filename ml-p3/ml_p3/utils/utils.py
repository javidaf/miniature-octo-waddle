import numpy as np
from ml_p3.feature_extraction import extract_features
from sklearn.model_selection import KFold
from ml_p2.neural_network import NeuralNetwork
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import os

import pickle


def prepare_cell_data(dataset_numbers, columns=None, verbose=True):
    train_features_list = []
    train_targets_list = []
    script_dir = os.path.dirname(__file__)

    for i in dataset_numbers:

        classifications_file = os.path.join(
            script_dir,
            "..",
            "data",
            "classification",
            f"classifications{i}.npy",
        )
        masks_file = os.path.join(
            script_dir,
            "..",
            "data",
            "classification",
            f"combined_mean_image{i}_seg.npy",
        )
        classifications_file = os.path.normpath(classifications_file)
        masks_file = os.path.normpath(masks_file)

        masks = np.load(masks_file, allow_pickle=True).item()["masks"]
        classifications = np.load(classifications_file, allow_pickle=True).item()[
            "classifications"
        ]

        features, targets = extract_features(masks, classifications)

        if columns is not None:
            features = features[:, columns]

        train_features_list.append(features)
        train_targets_list.append(targets)

    train_features = np.concatenate(train_features_list)
    train_targets = np.concatenate(train_targets_list)

    adjusted_train_targets = train_targets - 1

    num_classes = 3
    targets_one_hot = np.zeros((adjusted_train_targets.size, num_classes))
    targets_one_hot[np.arange(adjusted_train_targets.size), adjusted_train_targets] = 1

    if verbose:
        unique, counts = np.unique(train_targets, return_counts=True)
        total_samples = adjusted_train_targets.size
        print(f"Total samples: {total_samples}")
        for cls, count in zip(unique, counts):
            print(f"Class {int(cls)}: {count} samples")
        if columns is not None:
            print(f"Selected feature columns: {columns}")

    return train_features, adjusted_train_targets, targets_one_hot


def grid_search_nn(
    train_features,
    targets_one_hot,
    search_params=None,
    fixed_params=None,
    k_folds=5,
    epochs=200,
    random_state=42,
):
    """
    Flexible grid search for neural network parameters

    Args:
        search_params: Dict of parameters to search over, e.g.
            {'hidden_layers': [[3], [4]], 'activations': ['relu', 'sigmoid']}
        fixed_params: Dict of fixed parameters to use
    """
    default_params = {
        "hidden_layers": [[3]],
        "activations": ["relu"],
        "learning_rate": 0.01,
        "optimizer": "adam",
        "output_activation": "softmax",
        "use_regularization": False,
        "lambda_": 0.01,
        "classification_type": "multiclass",
        "initializer": "he",
    }

    if fixed_params:
        default_params.update(fixed_params)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    accuracies = []

    search_params = search_params or {}
    param_names = list(search_params.keys())
    param_values = list(search_params.values())

    for params in product(*param_values):
        current_params = default_params.copy()
        current_params.update(dict(zip(param_names, params)))

        fold_accuracies = []

        for train_idx, val_idx in kf.split(train_features):
            X_train_fold = train_features[train_idx]
            y_train_fold = targets_one_hot[train_idx]
            X_val_fold = train_features[val_idx]
            y_val_fold = targets_one_hot[val_idx]

            NN = NeuralNetwork(
                input_size=train_features.shape[1],
                hidden_layers=current_params["hidden_layers"],
                output_size=3,
                learning_rate=current_params["learning_rate"],
                optimizer=current_params["optimizer"],
                output_activation=current_params["output_activation"],
                hidden_activation=current_params["activations"],
                use_regularization=current_params["use_regularization"],
                lambda_=current_params["lambda_"],
                classification_type=current_params["classification_type"],
                initializer=current_params["initializer"],
            )

            NN.train_classifier(X_train_fold, y_train_fold, epochs=epochs)
            fold_accuracies.append(NN.accuracy_score(X_val_fold, y_val_fold))

        accuracies.append(np.mean(fold_accuracies))

    return accuracies, param_names, param_values


def save_neural_network(model, filepath):
    """Save neural network model to file"""
    model_state = {
        "params": model.params,
        "weights": model.weights,
        "biases": model.biases,
    }
    with open(filepath, "wb") as f:
        pickle.dump(model_state, f)


def load_neural_network(filepath):
    """Load neural network model from file"""
    with open(filepath, "rb") as f:
        model_state = pickle.load(f)

    # Recreate the model using the stored parameters
    model = NeuralNetwork(**model_state["params"])
    model.weights = model_state["weights"]
    model.biases = model_state["biases"]
    return model


def prepare_eth_data(csv_file, lookback=50, horizon=1, freq="D", split_ratio=0.8):
    """
    Prepare the data for the LSTM network with MinMax scaling.

    Parameters:
    - csv_file: str, path to the CSV file.
    - lookback: int, how many past time steps to use as input features.
    - horizon: int, how many future steps to predict.
    - freq: str, resampling frequency (e.g., 'D' for daily, 'H' for hourly, 'W' for weekly).
    - split_ratio: float, fraction of data to use for training.

    Returns:
    - X_train: np.ndarray, shape (train_samples, lookback, num_features)
    - Y_train: np.ndarray, shape (train_samples, horizon, 1)
    - X_test: np.ndarray, shape (test_samples, lookback, num_features)
    - Y_test: np.ndarray, shape (test_samples, horizon, 1)
    - df_resampled: pd.DataFrame, the resampled dataframe used for modeling
    """

    df = pd.read_csv(csv_file)

    # Convert 'Open time' to datetime and set as index
    df["Open time"] = pd.to_datetime(df["Open time"])
    df.set_index("Open time", inplace=True)

    # Resample data to desired frequency
    df_resampled = df.resample(freq).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )

    df_resampled.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Volume"]
    data = df_resampled[features].values

    split = int(len(data) * split_ratio)
    data_train = data[:split]
    data_test = data[split:]

    scaler = MinMaxScaler()
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # Create sequences for training data
    X_train = []
    Y_train = []
    for i in range(len(data_train_scaled) - lookback - horizon + 1):
        X_train.append(data_train_scaled[i : i + lookback])
        Y_train.append(
            data_train_scaled[i + lookback : i + lookback + horizon, 3]
        )  # Close price

    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape(-1, horizon, 1)

    # Create sequences for testing data
    X_test = []
    Y_test = []
    for i in range(len(data_test_scaled) - lookback - horizon + 1):
        X_test.append(data_test_scaled[i : i + lookback])
        Y_test.append(
            data_test_scaled[i + lookback : i + lookback + horizon, 3]
        )  # Close price

    X_test = np.array(X_test)
    Y_test = np.array(Y_test).reshape(-1, horizon, 1)

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

    return X_train, Y_train, X_test, Y_test, df_resampled, scaler


def load_lstm_model(filepath, LSTMNetwork):
    """Load a saved LSTMNetwork model from a pickle file."""
    with open(filepath, "rb") as f:
        model_state = pickle.load(f)

    optimizer = model_state["optimizer"]
    model = LSTMNetwork(
        input_dim=model_state["input_dim"],
        hidden_dim=model_state["hidden_dim"],
        output_dim=model_state["output_dim"],
        future_steps=model_state["future_steps"],
        optimizer=optimizer,
        initializer=model_state["initializer"],
    )

    model.W = model_state["W"]
    model.U = model_state["U"]
    model.b = model_state["b"]
    model.Wy = model_state["Wy"]
    model.by = model_state["by"]

    return model
