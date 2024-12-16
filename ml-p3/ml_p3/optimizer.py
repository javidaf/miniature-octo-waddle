from ml_p2.neural_network.optimizer import Optimizer
import numpy as np


class AdamLSTM(Optimizer):
    """Adam optimizer"""

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def initialize(self, weights, biases):
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, gradients_w, gradients_b):
        self.t += 1

        # Update weights
        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (
                gradients_w[i] ** 2
            )
            # Compute bias-corrected first moment estimate
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            # Update weights
            weights[i] -= (
                self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            )

        # Update biases
        for i in range(len(biases)):
            # Update biased first moment estimate
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
            # Update biased second raw moment estimate
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (
                gradients_b[i] ** 2
            )
            # Compute bias-corrected first moment estimate
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)
            # Update biases
            biases[i] -= (
                self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            )

        return weights, biases
