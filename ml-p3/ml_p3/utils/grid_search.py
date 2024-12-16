from ml_p3.LSTM import LSTMNetwork
from ml_p3.kerasLSTM import build_model, evaluate_model, prepare_data_for_keras
from ml_p3.optimizer import AdamLSTM
from sklearn.model_selection import train_test_split


def grid_search_lstm(
    data_file_path, param_grid, epochs=50, batch_size=32, learning_rate=0.01
):
    import itertools

    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    for values in combinations:
        params = dict(zip(param_names, values))
        lookback = params["lookback"]
        hidden_dim = params["hidden_dim"]
        print(f"Training with parameters: {params}")
        from ml_p3.utils.utils import prepare_eth_data

        X_train, Y_train, X_test, Y_test, df_resampled, scaler = prepare_eth_data(
            csv_file=data_file_path,
            lookback=lookback,
            horizon=1,
            freq="D",
            split_ratio=0.8,
        )

        optimizer = AdamLSTM(learning_rate=learning_rate)
        model = LSTMNetwork(
            input_dim=X_train.shape[-1],
            hidden_dim=hidden_dim,
            output_dim=1,
            future_steps=1,
            optimizer=optimizer,
        )

        model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        val_scores = model.scores(X_test, Y_test)

        result = {"params": params, "val_scores": val_scores}
        results.append(result)

    return results


def grid_search_keras_lstm(data_file_path, param_grid, epochs=50, batch_size=32):
    import itertools

    results = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    for values in combinations:
        params = dict(zip(param_names, values))
        lookback = params["lookback"]
        hidden_dim = params["hidden_dim"]
        print(f"Training with parameters: {params}")

        from ml_p3.utils.utils import prepare_eth_data

        X_train, Y_train, X_test, Y_test, _, _ = prepare_eth_data(
            csv_file=data_file_path,
            lookback=lookback,
            horizon=1,
            freq="D",
            split_ratio=0.8,
        )

        Y_train, Y_test = prepare_data_for_keras(Y_train, Y_test)

        input_dim = X_train.shape[-1]
        model = build_model(
            lookback=lookback,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            learning_rate=0.01,
        )

        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            verbose=1,
        )

        Y_val_pred, val_scores = evaluate_model(
            model, X_test, Y_test, data_type="Validation"
        )

        result = {
            "params": params,
            "val_scores": val_scores,
            "history": history.history,
        }
        results.append(result)

    return results
