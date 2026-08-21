import numpy as np

from loss import CrossEntropyLoss


def train_model(
    gpt,
    embedding_inputs,
    target_outputs,
    num_epochs=15,
    learning_rate=1e-3,
    start_epoch=0,
    checkpoint_prefix="gpt_stride4_epoch"
):

    criterion = CrossEntropyLoss()

    for epoch in range(
        start_epoch,
        start_epoch + num_epochs
    ):

        total_loss = 0.0

        for batch_idx in range(
            len(embedding_inputs)
        ):

            # ==========================================
            # Token IDs
            # ==========================================

            token_batch = np.asarray(
                embedding_inputs[batch_idx],
                dtype=np.int64
            )

            train_targets = np.asarray(
                target_outputs[batch_idx],
                dtype=np.int64
            )

            # ==========================================
            # Current GPT token embeddings
            # ==========================================

            train_inputs = (
                gpt.embedding_matrix[
                    token_batch
                ]
            )

            # ==========================================
            # Add GPT positional embeddings
            # ==========================================

            train_inputs = (
                train_inputs
                + gpt.position_embeddings
            )

            # ==========================================
            # Add outer dimensions
            # ==========================================

            train_inputs = train_inputs[
                np.newaxis,
                :,
                :,
                :
            ]

            train_targets = train_targets[
                np.newaxis,
                :,
                :
            ]

            # ==========================================
            # Forward
            # ==========================================

            logits = gpt.forward(
                train_inputs
            )

            # ==========================================
            # Loss
            # ==========================================

            loss = criterion.forward(
                logits,
                train_targets
            )

            total_loss += loss

            # ==========================================
            # Backward
            # ==========================================

            d_logits = criterion.backward()

            gpt.backward(
                d_logits,
                learning_rate=learning_rate
            )

            # ==========================================
            # Progress
            # ==========================================

            if batch_idx % 500 == 0:

                print(
                    "Epoch:",
                    epoch + 1,
                    "Batch:",
                    batch_idx,
                    "Loss:",
                    loss
                )

        # ==============================================
        # Epoch average loss
        # ==============================================

        average_loss = (
            total_loss
            / len(embedding_inputs)
        )

        print()
        print(
            "Epoch:",
            epoch + 1,
            "Average Loss:",
            average_loss
        )
        print()

        # ==============================================
        # Save checkpoint
        # ==============================================

        filename = (
            f"{checkpoint_prefix}_"
            f"{epoch + 1}.pkl"
        )

        gpt.save_weights(
            filename
        )

    return gpt