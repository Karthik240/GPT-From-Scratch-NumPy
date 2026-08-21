import numpy as np

from layers import LayerNormalization, FeedForward
from attention import MultiHeadAttention

class TransformerBlock:

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        sequence_length=16
    ):

        self.ln1 = LayerNormalization()

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            sequence_length
        )

        self.ln2 = LayerNormalization()

        self.ffn = FeedForward(embed_dim)

    def forward(self, batches):

        output = []

        for batch in batches:

            batch_out = []

            for seq in batch:

                # ============================
                # Attention
                # ============================

                residual = seq

                x = self.ln1.forward(seq)

                x = self.attention.forward(x)

                # Residual connection
                x = x + residual

                # ============================
                # Feed Forward
                # ============================

                residual = x

                y = self.ln2.forward(x)

                y = self.ffn.forward(y)

                # Residual connection
                y = y + residual

                batch_out.append(y)

            output.append(batch_out)

        return np.array(output)

    def backward(
        self,
        dout,
        learning_rate=1e-3
    ):

        output = []

        # IMPORTANT:
        #
        # Forward order:
        #   batch 0
        #      seq 0
        #      seq 1
        #      seq 2
        #
        # Backward must be:
        #   batch 0
        #      seq 2
        #      seq 1
        #      seq 0
        #
        # because all our layer caches use pop().

        for batch in reversed(dout):

            batch_grad = []

            for grad in reversed(batch):

                # ============================
                # Feed Forward residual
                # ============================

                residual_grad = grad

                grad = self.ffn.backward(
                    grad,
                    learning_rate
                )

                grad = self.ln2.backward(grad)

                # Residual gradient
                grad = grad + residual_grad

                # ============================
                # Attention residual
                # ============================

                residual_grad = grad

                grad = self.attention.backward(
                    grad,
                    learning_rate
                )

                grad = self.ln1.backward(grad)

                # Residual gradient
                grad = grad + residual_grad

                batch_grad.append(grad)

            # Restore original sequence order
            batch_grad.reverse()

            output.append(batch_grad)

        # Restore original batch order
        output.reverse()

        return np.array(output)