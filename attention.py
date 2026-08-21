import numpy as np
class SelfAttentionHead:

    def __init__(self, embed_dim, head_dim, sequence_length):

        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.Wq = np.random.randn(
            embed_dim,
            head_dim
        ) * 0.02

        self.Wk = np.random.randn(
            embed_dim,
            head_dim
        ) * 0.02

        self.Wv = np.random.randn(
            embed_dim,
            head_dim
        ) * 0.02

        # Causal mask
        self.mask = np.triu(
            np.ones(
                (sequence_length, sequence_length),
                dtype=bool
            ),
            k=1
        )

        self.cache = []

    def forward(self, seq):

        Q = seq @ self.Wq
        K = seq @ self.Wk
        V = seq @ self.Wv
    
        # ==========================================
        # Attention scores
        # ==========================================
    
        scale = np.sqrt(self.head_dim)
    
        scores = (Q @ K.T) / scale
    
        # ==========================================
        # Dynamic causal mask
        # ==========================================
    
        seq_len = seq.shape[0]
    
        mask = self.mask[:seq_len, :seq_len]
    
        scores[mask] = -1e9
    
        # ==========================================
        # Stable softmax
        # ==========================================
    
        scores = (
            scores
            - np.max(
                scores,
                axis=1,
                keepdims=True
            )
        )
    
        exp_scores = np.exp(scores)
    
        attn = (
            exp_scores
            / np.sum(
                exp_scores,
                axis=1,
                keepdims=True
            )
        )
    
        # ==========================================
        # Context
        # ==========================================
    
        context = attn @ V
    
        # ==========================================
        # Cache
        # ==========================================
    
        self.cache.append(
            (
                seq,
                Q,
                K,
                V,
                attn
            )
        )
    
        return context

    def backward(self, dcontext, learning_rate=1e-3):

        # Because we use pop(),
        # TransformerBlock.backward()
        # must process sequences in reverse order.

        seq, Q, K, V, attn = self.cache.pop()

        scale = np.sqrt(self.head_dim)

        # ============================
        # Context = attn @ V
        # ============================

        d_attn = dcontext @ V.T

        dV = attn.T @ dcontext

        # ============================
        # V weights
        # ============================

        dWv = seq.T @ dV

        dx_v = dV @ self.Wv.T

        # ============================
        # Softmax backward
        # ============================

        d_scores = np.zeros_like(attn)

        for i in range(attn.shape[0]):

            a = attn[i]
            da = d_attn[i]

            J = np.diag(a) - np.outer(a, a)

            d_scores[i] = J @ da

        # Do not propagate gradients
        # through masked positions
        d_scores[self.mask] = 0

        # ============================
        # Q and K
        # ============================

        dQ = (d_scores @ K) / scale

        dK = (d_scores.T @ Q) / scale

        # ============================
        # Q/K weights
        # ============================

        dWq = seq.T @ dQ
        dWk = seq.T @ dK

        dx_q = dQ @ self.Wq.T
        dx_k = dK @ self.Wk.T

        # ============================
        # Update weights
        # ============================

        self.Wq -= learning_rate * dWq

        self.Wk -= learning_rate * dWk

        self.Wv -= learning_rate * dWv

        # ============================
        # Gradient to input
        # ============================

        dx = dx_q + dx_k + dx_v

        return dx


class MultiHeadAttention:

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        sequence_length=16
    ):

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = [
            SelfAttentionHead(
                embed_dim,
                self.head_dim,
                sequence_length
            )
            for _ in range(num_heads)
        ]

        # Output projection
        self.Wo = np.random.randn(
            embed_dim,
            embed_dim
        ) * 0.02

        self.cache = []

    def forward(self, seq):

        head_outputs = []

        # ============================
        # Run every attention head
        # ============================

        for head in self.heads:

            head_outputs.append(
                head.forward(seq)
            )

        # ============================
        # Concatenate heads
        # ============================

        concat = np.concatenate(
            head_outputs,
            axis=1
        )

        # Save for output projection backward
        self.cache.append(concat)

        # ============================
        # Output projection
        # ============================

        out = concat @ self.Wo

        return out

    def backward(
        self,
        dout,
        learning_rate=1e-3
    ):

        # Must correspond to the same sequence
        # whose gradient is currently being processed.
        concat = self.cache.pop()

        # ============================
        # Output projection
        # ============================

        dWo = concat.T @ dout

        dconcat = dout @ self.Wo.T

        # ============================
        # Update Wo
        # ============================

        self.Wo -= (
            learning_rate * dWo
        )

        # ============================
        # Split gradient
        # ============================

        split_grads = np.split(
            dconcat,
            self.num_heads,
            axis=1
        )

        # ============================
        # Gradient to input
        # ============================

        dx = np.zeros(
            (
                concat.shape[0],
                self.embed_dim
            )
        )

        # Each head receives its own
        # portion of the concatenated gradient
        for grad, head in zip(
            split_grads,
            self.heads
        ):

            dx += head.backward(
                grad,
                learning_rate
            )

        return dx