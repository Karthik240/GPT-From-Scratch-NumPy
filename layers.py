import numpy as np
class LayerNormalization:

    def __init__(self, eps=1e-5):
        self.eps = eps
        self.cache = []

    def forward(self, seq):

        if seq.ndim != 2:
            raise ValueError(
                f"LayerNormalization expected (sequence_length, embed_dim), "
                f"but received {seq.shape}"
            )

        mean = np.mean(
            seq,
            axis=1,
            keepdims=True
        )

        variance = np.mean(
            (seq - mean) ** 2,
            axis=1,
            keepdims=True
        )

        std = np.sqrt(
            variance + self.eps
        )

        x_hat = (seq - mean) / std

        self.cache.append(
            (
                seq,
                mean,
                variance,
                std,
                x_hat
            )
        )

        return x_hat

    def backward(self, dout):

        x, mean, variance, std, x_hat = self.cache.pop()

        D = x.shape[1]

        dx = (
            (1.0 / D)
            * (1.0 / std)
            * (
                D * dout
                - np.sum(
                    dout,
                    axis=1,
                    keepdims=True
                )
                - x_hat * np.sum(
                    dout * x_hat,
                    axis=1,
                    keepdims=True
                )
            )
        )

        return dx

class GELU:
    def __init__(self):

        self.cache = []

    def forward(self, x):

        t = np.sqrt(2/np.pi) * (
            x + 0.044715 * (x**3)
        )

        tanh = np.tanh(t)

        self.cache.append((x, tanh))

        return 0.5 * x * (1 + tanh)

    def backward(self, dout):

        x, tanh = self.cache.pop()

        c = np.sqrt(2/np.pi)

        dt_dx = c * (
            1 + 3 * 0.044715 * x**2
        )

        dy_dx = (
            0.5 * (1 + tanh)
            +
            0.5 * x * (1 - tanh**2) * dt_dx
        )

        dx = dout * dy_dx

        return dx

class FeedForward:
    def __init__(self, embed_dim):

        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * 4

        self.W1 = np.random.randn(
            embed_dim,
            self.hidden_dim
        ) * 0.02

        self.W2 = np.random.randn(
            self.hidden_dim,
            embed_dim
        ) * 0.02

        self.gelu = GELU()

        self.cache = []

    def forward(self, seq):

        linear1 = seq @ self.W1

        gelu_out = self.gelu.forward(linear1)

        output = gelu_out @ self.W2

        self.cache.append(
            (
                seq,
                gelu_out
            )
        )

        return output

    def backward(self, dout, learning_rate=1e-3):

        # IMPORTANT:
        # pop() means backward must process
        # sequences in reverse order.

        input, gelu_out = self.cache.pop()

        # -------------------------------
        # Linear 2
        # -------------------------------

        dW2 = gelu_out.T @ dout

        dgelu = dout @ self.W2.T

        # -------------------------------
        # GELU
        # -------------------------------

        dlinear1 = self.gelu.backward(dgelu)

        # -------------------------------
        # Linear 1
        # -------------------------------

        dW1 = input.T @ dlinear1

        dx = dlinear1 @ self.W1.T

        # -------------------------------
        # Update weights
        # -------------------------------

        self.W2 -= learning_rate * dW2

        self.W1 -= learning_rate * dW1

        return dx