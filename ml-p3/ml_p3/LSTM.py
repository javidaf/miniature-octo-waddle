import pickle
from matplotlib import pyplot as plt
import numpy as np
from ml_p2.neural_network.base import BaseNeuralNetwork
from ml_p2.neural_network.activations import Activation
from ml_p3.optimizer import AdamLSTM
from ml_p2.neural_network.initializers import Initializer


class LSTMNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        future_steps=1,
        optimizer=None,
        initializer=None,
    ):
        """
        input_dim: number of input features
        hidden_dim: number of units in the LSTM cell
        output_dim: number of output features (e.g. 1 for a single price or more if multiple)
        future_steps: how many steps in the future to predict
        optimizer: optimizer instance (e.g. Adam, SGD, etc.)
        initializer: initialization method for weights
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.future_steps = future_steps
        self.optimizer = optimizer
        self.initializer = (
            initializer if initializer is not None else self._default_initializer
        )

        # LSTM parameters
        # The LSTM cell: we need to create parameters for four gates:
        # f (forget), i (input), o (output), and g (candidate cell).
        # We'll combine them into a single matrix to simplify.
        # Dimensions:
        #   Wf, Wi, Wc, Wo: (input_dim, hidden_dim)
        #   Uf, Ui, Uc, Uo: (hidden_dim, hidden_dim)
        #   bf, bi, bc, bo: (hidden_dim,)

        # We'll stack them into bigger matrices for efficiency:
        # W: shape (input_dim, 4*hidden_dim)
        # U: shape (hidden_dim, 4*hidden_dim)
        # b: shape (4*hidden_dim, )

        self.W = self.initializer(self.input_dim, 4 * self.hidden_dim)
        self.U = self.initializer(self.hidden_dim, 4 * self.hidden_dim)
        self.b = np.zeros((4 * self.hidden_dim,))

        # Output layer parameters (from h_t to output)
        # We will produce all future_steps at once from the final hidden state or last time step's hidden state.
        self.Wy = self.initializer(self.hidden_dim, self.output_dim * self.future_steps)
        self.by = np.zeros((self.output_dim * self.future_steps,))

        # For the optimizer
        if self.optimizer is not None:
            self.optimizer.initialize([self.W, self.U, self.Wy], [self.b, self.by])

    def _default_initializer(self, input_dim, output_dim):
        return Initializer.xavier(input_dim, output_dim)

    def _forward_lstm(self, X):
        """
        Forward pass through LSTM for the entire sequence.
        X shape: (batch_size, seq_len, input_dim)
        Returns:
            hs: all hidden states (batch_size, seq_len, hidden_dim)
            cs: all cell states (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = X.shape

        hs = np.zeros((batch_size, seq_len, self.hidden_dim))
        cs = np.zeros((batch_size, seq_len, self.hidden_dim))

        h_t = np.zeros((batch_size, self.hidden_dim))
        c_t = np.zeros((batch_size, self.hidden_dim))

        self.cache = []

        for t in range(seq_len):
            x_t = X[:, t, :]

            z = x_t @ self.W + h_t @ self.U + self.b  # (batch_size, 4*hidden_dim)
            f_t = Activation.sigmoid(
                z[:, 0 * self.hidden_dim : 1 * self.hidden_dim]
            )  # forget gate
            i_t = Activation.sigmoid(
                z[:, 1 * self.hidden_dim : 2 * self.hidden_dim]
            )  # input gate
            g_t = np.tanh(z[:, 2 * self.hidden_dim : 3 * self.hidden_dim])  # candidate
            o_t = Activation.sigmoid(
                z[:, 3 * self.hidden_dim : 4 * self.hidden_dim]
            )  # output gate

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            hs[:, t, :] = h_t
            cs[:, t, :] = c_t

            self.cache.append((x_t, h_t, c_t, f_t, i_t, g_t, o_t))

        return hs, cs

    def forward(self, X):
        """
        Forward pass for the entire batch.
        X shape: (batch_size, seq_len, input_dim)
        Returns:
            Y_pred: (batch_size, future_steps, output_dim)
        """
        hs, cs = self._forward_lstm(X)

        # Use the last hidden state to predict the future steps.
        h_last = hs[:, -1, :]
        Y_pred = h_last @ self.Wy + self.by  # (batch_size, future_steps * output_dim)
        # Reshape into (batch_size, future_steps, output_dim)
        Y_pred = Y_pred.reshape(-1, self.future_steps, self.output_dim)
        return Y_pred

    def backward(self, dY, X):
        """
        Backward pass through the LSTM and output layer.
        dY: gradient w.r.t output Y_pred: (batch_size, future_steps, output_dim)
        X: input (batch_size, seq_len, input_dim)

        BPTT through the LSTM.
        """
        batch_size, seq_len, _ = X.shape

        # Reshape dY to match Wy shape
        dY = dY.reshape(batch_size, self.future_steps * self.output_dim)

        # Gradients for output layer
        dWy = 0
        dby = 0

        # LSTM gradients
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)

        dh_t = dY @ self.Wy.T  # gradient w.r.t. last hidden state
        dc_t = np.zeros((batch_size, self.hidden_dim))

        dWy = self.cache[-1][1].T @ dY  # h_last from last time step
        dby = np.sum(dY, axis=0)

        # BPTT through LSTM
        for t in reversed(range(seq_len)):
            x_t, h_t, c_t, f_t, i_t, g_t, o_t = self.cache[t]
            # h_t at time t is cached in hs, but we have it here as well.

            # dh_t and dc_t from the future (initially from output layer)
            do_t = dh_t * np.tanh(c_t)
            dc_t = dc_t + (dh_t * o_t * (1 - np.tanh(c_t) ** 2))

            if t > 0:
                c_prev = self.cache[t - 1][2]  # c_{t-1}
            else:
                c_prev = np.zeros_like(c_t)

            df_t = dc_t * c_prev
            di_t = dc_t * g_t
            dg_t = dc_t * i_t
            # c_{t-1} is c_t at previous step. If t=0, c_{t-1} is zero.
            c_prev = self.cache[t - 1][2] if t > 0 else np.zeros_like(c_t)

            # gate derivatives
            do_t = do_t * o_t * (1 - o_t)
            df_t = df_t * f_t * (1 - f_t)
            di_t = di_t * i_t * (1 - i_t)
            dg_t = dg_t * (1 - g_t**2)  # derivative of tanh

            # Combine
            dz = np.hstack((df_t, di_t, dg_t, do_t))  # (batch_size, 4*hidden_dim)

            dW += x_t.T @ dz
            dU += self.cache[t - 1][1].T @ dz if t > 0 else np.zeros_like(self.U)
            db += np.sum(dz, axis=0)

            # dX, dH_t-1, dC_t-1 for next iteration
            dx_t = dz @ self.W.T
            dh_prev = dz @ self.U.T
            dc_t = dc_t * f_t

            dh_t = dh_prev  # carry this back in time

        # Average gradients by batch size
        dW /= batch_size
        dU /= batch_size
        db /= batch_size
        dWy /= batch_size
        dby /= batch_size

        return dW, dU, db, dWy, dby

    def train(self, X, y, epochs=10, batch_size=32):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            total_loss = 0.0
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                Y_pred = self.forward(X_batch)
                loss = np.mean((Y_pred - y_batch) ** 2)
                total_loss += loss * len(X_batch)

                dY = 2 * (Y_pred - y_batch) / y_batch.size
                dW, dU, db, dWy, dby = self.backward(dY, X_batch)

                # Update
                weights = [self.W, self.U, self.Wy]
                biases = [self.b, self.by]
                gradients_w = [dW, dU, dWy]
                gradients_b = [db, dby]

                self.W, self.U, self.Wy, self.b, self.by = self._update_params(
                    weights, biases, gradients_w, gradients_b
                )

            # Average epoch loss
            avg_epoch_loss = total_loss / n_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

    def _update_params(self, weights, biases, gradients_w, gradients_b):
        if self.optimizer is not None:
            updated_w, updated_b = self.optimizer.update(
                weights, biases, gradients_w, gradients_b
            )
            return updated_w[0], updated_w[1], updated_w[2], updated_b[0], updated_b[1]
        else:
            # Simple SGD if no optimizer provided
            lr = 0.01
            for i in range(len(weights)):
                weights[i] -= lr * gradients_w[i]
                biases[i] -= lr * gradients_b[i]
            return weights[0], weights[1], weights[2], biases[0], biases[1]

    def predict(self, X):
        """
        Predict future steps based on input sequence X.
        """
        return self.forward(X)

    def save(self, filepath):
        """Save the model to a file"""
        model_state = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "future_steps": self.future_steps,
            "optimizer": self.optimizer,
            "initializer": self.initializer,
            "W": self.W,
            "U": self.U,
            "b": self.b,
            "Wy": self.Wy,
            "by": self.by,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_state, f)

    def scores(self, X, y):
        """Return the loss and accuracy"""
        Y_pred = self.predict(X)
        scores_dict = {}
        # MAE
        scores_dict["mae"] = np.mean(np.abs(Y_pred - y))
        # MSE
        scores_dict["mse"] = np.mean((Y_pred - y) ** 2)
        # RMSE
        scores_dict["rmse"] = np.sqrt(scores_dict["mse"])

        # R2
        numerator = np.sum((y - Y_pred) ** 2)
        denominator = np.sum((y - np.mean(y)) ** 2)
        scores_dict["r2"] = 1 - numerator / denominator

        return scores_dict


def main():
    input_dim = 5  # 5 features: open, high, low, close, volume
    hidden_dim = 16
    output_dim = 1  # predict one price (closing price)
    future_steps = 1
    lookback = 2

    optimizer = AdamLSTM(learning_rate=0.01)
    model = LSTMNetwork(
        input_dim, hidden_dim, output_dim, future_steps, optimizer=optimizer
    )

    # X_train: (batch_size, seq_len, input_dim)
    # y_train: (batch_size, future_steps, output_dim)
    from ml_p3.utils.utils import prepare_eth_data

    data_file_path = r"ml-p3\tests\data\ETHUSD_1m_Binance.csv"
    X_train, Y_train, X_test, Y_test, df_resampled, scale = prepare_eth_data(
        data_file_path,
        lookback=lookback,
        horizon=future_steps,
        freq="D",  # daily aggregation
        split_ratio=0.8,
    )

    model.train(X_train, Y_train, epochs=50, batch_size=32)
    y_train_scores = model.scores(X_train, Y_train)
    print("Training scores:", y_train_scores)

    y_pred = model.predict(X_test)
    y_test_scores = model.scores(X_test, Y_test)
    print("Testing scores:", y_test_scores)
    plt.plot(Y_test.flatten(), label="Actual")
    plt.plot(y_pred.flatten(), label="Predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
