import numpy as np
import pickle

from transformer import TransformerBlock
from layers import LayerNormalization

import numpy as np
import pickle


class GPT:

    def __init__(
        self,
        num_layers=1,
        embed_dim=256,
        num_heads=4,
        sequence_length=16,
        embedding_matrix=None
    ):

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        # ==========================================
        # Input embedding matrix
        # ==========================================

        if embedding_matrix is None:
            raise ValueError(
                "embedding_matrix must be provided"
            )

        self.embedding_matrix = embedding_matrix.copy()

        self.vocab_size = embedding_matrix.shape[0]
        self.position_embeddings = (
            np.random.randn(
                sequence_length,
                embed_dim
            ) * 0.02
        )
        # ==========================================
        # Separate output projection
        #
        # (embed_dim, vocab_size)
        # ==========================================

        self.W_out = (
            np.random.randn(
                embed_dim,
                self.vocab_size
            ) * 0.02
        )

        # ==========================================
        # Transformer blocks
        # ==========================================

        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                sequence_length
            )
            for _ in range(num_layers)
        ]

        # ==========================================
        # Final LayerNorm
        # ==========================================

        self.final_ln = LayerNormalization()

        # Final hidden representation cache
        self.x = None


    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, x):

        # x:
        # (batch,
        #  sequence_count,
        #  sequence_length,
        #  embed_dim)

        for block in self.blocks:
            x = block.forward(x)

        # ==========================================
        # Final LayerNorm
        # ==========================================

        final = []

        for batch in x:

            temp = []

            for seq in batch:

                temp.append(
                    self.final_ln.forward(seq)
                )

            final.append(temp)

        x = np.array(final)

        # Save for backward
        self.x = x

        # ==========================================
        # Vocabulary projection
        #
        # x:
        # (..., embed_dim)
        #
        # W_out:
        # (embed_dim, vocab_size)
        #
        # logits:
        # (..., vocab_size)
        # ==========================================

        logits = x @ self.W_out

        return logits


    # ======================================================
    # BACKWARD
    # ======================================================

    def backward(
        self,
        d_logits,
        learning_rate=1e-3
    ):

        # ==========================================
        # Flatten
        # ==========================================

        x_flat = self.x.reshape(
            -1,
            self.x.shape[-1]
        )

        d_logits_flat = d_logits.reshape(
            -1,
            d_logits.shape[-1]
        )

        # ==========================================
        # Output projection gradient
        #
        # logits = x @ W_out
        # ==========================================

        dW_out = (
            x_flat.T @ d_logits_flat
        )

        # Gradient into final LayerNorm
        dx = (
            d_logits @ self.W_out.T
        )

        # ==========================================
        # Update W_out
        # ==========================================

        self.W_out -= (
            learning_rate * dW_out
        )

        # ==========================================
        # Final LayerNorm backward
        # ==========================================

        final_dx = np.zeros_like(dx)

        # Reverse because LayerNorm cache uses pop()
        for b in range(
            dx.shape[0] - 1,
            -1,
            -1
        ):

            for s in range(
                dx.shape[1] - 1,
                -1,
                -1
            ):

                final_dx[b, s] = (
                    self.final_ln.backward(
                        dx[b, s]
                    )
                )

        dx = final_dx

        # ==========================================
        # Transformer blocks backward
        # ==========================================

        for block in reversed(self.blocks):

            dx = block.backward(
                dx,
                learning_rate
            )

        return dx


    # ======================================================
    # SAVE WEIGHTS
    # ======================================================

    def save_weights(
        self,
        filename="gpt_weights.pkl"
    ):

        data = {

            "num_layers":
                self.num_layers,

            "embed_dim":
                self.embed_dim,

            "num_heads":
                self.num_heads,

            "sequence_length":
                self.sequence_length,

            "vocab_size":
                self.vocab_size,

            # Input embeddings
            "embedding_matrix":
                self.embedding_matrix.copy(),

            # Separate output projection
            "W_out":
                self.W_out.copy(),
            "position_embeddings":
                self.position_embeddings.copy(),
            "blocks": []
        }

        # ==========================================
        # Transformer blocks
        # ==========================================

        for block in self.blocks:

            block_data = {

                "attention_Wo":
                    block.attention.Wo.copy(),

                "attention_heads": [],

                "ffn_W1":
                    block.ffn.W1.copy(),

                "ffn_W2":
                    block.ffn.W2.copy()
            }

            # ======================================
            # Attention heads
            # ======================================

            for head in block.attention.heads:

                head_data = {

                    "Wq":
                        head.Wq.copy(),

                    "Wk":
                        head.Wk.copy(),

                    "Wv":
                        head.Wv.copy()
                }

                block_data[
                    "attention_heads"
                ].append(head_data)

            data["blocks"].append(
                block_data
            )

        # ==========================================
        # Write checkpoint
        # ==========================================

        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(
            "Weights saved to:",
            filename
        )


    # ======================================================
    # LOAD WEIGHTS
    # ======================================================

    def load_weights(
        self,
        filename="gpt_weights.pkl"
    ):

        with open(filename, "rb") as f:
            data = pickle.load(f)

        # ==========================================
        # Architecture checks
        # ==========================================

        if data["num_layers"] != self.num_layers:
            raise ValueError(
                "Number of layers does not match"
            )

        if data["embed_dim"] != self.embed_dim:
            raise ValueError(
                "Embedding dimension does not match"
            )

        if data["num_heads"] != self.num_heads:
            raise ValueError(
                "Number of heads does not match"
            )

        if (
            data["sequence_length"]
            != self.sequence_length
        ):
            raise ValueError(
                "Sequence length does not match"
            )

        if (
            data.get(
                "vocab_size",
                data["embedding_matrix"].shape[0]
            )
            != self.vocab_size
        ):
            raise ValueError(
                "Vocabulary size does not match"
            )

        # ==========================================
        # Input embedding matrix
        # ==========================================

        if (
            data["embedding_matrix"].shape
            != self.embedding_matrix.shape
        ):
            raise ValueError(
                "Embedding matrix shape does not match"
            )

        self.embedding_matrix = (
            data["embedding_matrix"].copy()
        )

        # ==========================================
        # Output projection
        # ==========================================

        if "W_out" not in data:

            raise ValueError(
                "This checkpoint was created before "
                "the separate W_out output layer was added."
            )

        if (
            data["W_out"].shape
            != self.W_out.shape
        ):
            raise ValueError(
                "W_out shape does not match"
            )

        self.W_out = (
            data["W_out"].copy()
        )
        global position_embeddings

        if "position_embeddings" not in data:
            raise ValueError(
                "Checkpoint does not contain position_embeddings"
            )
        if (
            data["position_embeddings"].shape
            != self.position_embeddings.shape
        ):
            raise ValueError(
                "Position embedding shape does not match"
            )
        self.position_embeddings = (
            data["position_embeddings"].copy()
        )

        # ==========================================
        # Transformer blocks
        # ==========================================

        if (
            len(data["blocks"])
            != len(self.blocks)
        ):
            raise ValueError(
                "Number of transformer blocks does not match"
            )

        for block, block_data in zip(
            self.blocks,
            data["blocks"]
        ):

            # --------------------------------------
            # Attention output projection
            # --------------------------------------

            if (
                block_data["attention_Wo"].shape
                != block.attention.Wo.shape
            ):
                raise ValueError(
                    "Attention Wo shape does not match"
                )

            block.attention.Wo = (
                block_data[
                    "attention_Wo"
                ].copy()
            )

            # --------------------------------------
            # Feed Forward
            # --------------------------------------

            if (
                block_data["ffn_W1"].shape
                != block.ffn.W1.shape
            ):
                raise ValueError(
                    "FFN W1 shape does not match"
                )

            if (
                block_data["ffn_W2"].shape
                != block.ffn.W2.shape
            ):
                raise ValueError(
                    "FFN W2 shape does not match"
                )

            block.ffn.W1 = (
                block_data[
                    "ffn_W1"
                ].copy()
            )

            block.ffn.W2 = (
                block_data[
                    "ffn_W2"
                ].copy()
            )

            # --------------------------------------
            # Attention heads
            # --------------------------------------

            if (
                len(
                    block_data[
                        "attention_heads"
                    ]
                )
                != len(
                    block.attention.heads
                )
            ):
                raise ValueError(
                    "Number of attention heads does not match"
                )

            for head, head_data in zip(
                block.attention.heads,
                block_data[
                    "attention_heads"
                ]
            ):

                head.Wq = (
                    head_data["Wq"].copy()
                )

                head.Wk = (
                    head_data["Wk"].copy()
                )

                head.Wv = (
                    head_data["Wv"].copy()
                )

        print(
            "Weights loaded from:",
            filename
        )