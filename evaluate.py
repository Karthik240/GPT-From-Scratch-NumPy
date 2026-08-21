import numpy as np


def evaluate_model(
    gpt,
    tokenizer,
    embedding_inputs,
    target_outputs,
    max_batches=None
):

    space_id = tokenizer.str_to_int[" "]

    correct = 0
    total = 0

    non_space_correct = 0
    non_space_total = 0

    space_predictions = 0

    # Decide how many batches to evaluate
    if max_batches is None:
        num_batches = len(embedding_inputs)
    else:
        num_batches = min(
            max_batches,
            len(embedding_inputs)
        )

    for batch_idx in range(num_batches):

        # ==========================================
        # Token IDs
        # ==========================================

        token_batch = np.asarray(
            embedding_inputs[batch_idx],
            dtype=np.int64
        )

        targets = np.asarray(
            target_outputs[batch_idx],
            dtype=np.int64
        )

        # ==========================================
        # Token embeddings
        # ==========================================

        inputs = gpt.embedding_matrix[
            token_batch
        ]

        # Add positional embeddings
        inputs = (
            inputs
            + gpt.position_embeddings
        )

        # Add outer dimensions
        inputs = inputs[
            np.newaxis,
            :,
            :,
            :
        ]

        targets = targets[
            np.newaxis,
            :,
            :
        ]

        # ==========================================
        # Forward
        # ==========================================

        logits = gpt.forward(
            inputs
        )

        predictions = np.argmax(
            logits,
            axis=-1
        )

        # ==========================================
        # Overall accuracy
        # ==========================================

        correct += np.sum(
            predictions == targets
        )

        total += targets.size

        # ==========================================
        # Non-space accuracy
        # ==========================================

        mask = (
            targets != space_id
        )

        non_space_correct += np.sum(
            (predictions == targets)
            & mask
        )

        non_space_total += np.sum(
            mask
        )

        # ==========================================
        # Space prediction percentage
        # ==========================================

        space_predictions += np.sum(
            predictions == space_id
        )

        # ==========================================
        # Clear caches
        # ==========================================

        gpt.final_ln.cache.clear()

        for block in gpt.blocks:

            block.ln1.cache.clear()
            block.ln2.cache.clear()

            block.attention.cache.clear()

            for head in block.attention.heads:
                head.cache.clear()

            block.ffn.cache.clear()
            block.ffn.gelu.cache.clear()

    # ==============================================
    # Final metrics
    # ==============================================

    overall_accuracy = (
        correct / total
    )

    non_space_accuracy = (
        non_space_correct
        / non_space_total
    )

    space_prediction_percentage = (
        space_predictions
        / total
    )

    return {
        "overall_accuracy":
            overall_accuracy,

        "non_space_accuracy":
            non_space_accuracy,

        "space_prediction_percentage":
            space_prediction_percentage
    }