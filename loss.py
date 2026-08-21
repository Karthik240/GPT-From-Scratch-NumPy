import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.loss = None

    def forward(self, logits, targets):
        """
        logits  : (batchs, batch_size, seq_len, vocab_size)
        targets : (batchs, batch_size, seq_len)
        """

        # Stable Softmax
        logits = logits - np.max(logits, axis=-1, keepdims=True)

        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)

        # Save probabilities (needed for backprop later)
        self.probs = probs
        self.targets = targets.astype(np.int64)

        # Pick probability of correct class
        correct_probs = np.take_along_axis(
            probs,
            targets[..., np.newaxis],
            axis=-1
        ).squeeze(-1)

        # Cross Entropy
        loss = -np.log(correct_probs + 1e-9)

        # Average over every token
        self.loss = np.mean(loss)

        return self.loss
    def backward(self):

        d_logits = self.probs.copy()
        B, S, L = self.targets.shape
        batch_idx = np.arange(B)[:, None, None]
        seq_idx   = np.arange(S)[None, :, None]
        token_idx = np.arange(L)[None, None, :]
        d_logits[batch_idx,
                 seq_idx,
                 token_idx,
                 self.targets] -= 1
    
        d_logits /= (B * S * L)
    
        return d_logits