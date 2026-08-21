import numpy as np
def generate(
    gpt,
    tokenizer,
    prompt,
    max_new_tokens=20,
    temperature=0.8,
    do_sample=True,
    top_k=10
):

    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):

        # ==========================================
        # Keep only model context
        # ==========================================

        context_ids = token_ids[
            -gpt.sequence_length:
        ]

        # ==========================================
        # Token IDs -> embeddings
        # ==========================================

        x = gpt.embedding_matrix[
            np.asarray(
                context_ids,
                dtype=np.int64
            )
        ]

        # ==========================================
        # Add positional embeddings
        # ==========================================

        x = (
            x
            + gpt.position_embeddings[
                :len(context_ids)
            ]
        )

        # ==========================================
        # Add outer dimensions
        # ==========================================

        x = x[
            np.newaxis,
            np.newaxis,
            :,
            :
        ]

        # ==========================================
        # Forward
        # ==========================================

        logits = gpt.forward(x)

        next_logits = (
            logits[0, 0, -1].copy()
        )

        # ==========================================
        # Never generate <unk>
        # ==========================================

        unk_id = tokenizer.str_to_int["<unk>"]

        next_logits[unk_id] = -np.inf

        # ==========================================
        # Greedy decoding
        # ==========================================

        if not do_sample:

            next_token = np.argmax(
                next_logits
            )

        # ==========================================
        # Top-k Sampling
        # ==========================================

        else:

            if temperature <= 0:
                raise ValueError(
                    "temperature must be greater than 0"
                )

            # ------------------------------
            # Select top-k tokens only
            # ------------------------------

            k = min(
                top_k,
                len(next_logits)
            )

            top_ids = np.argsort(
                next_logits
            )[-k:]

            top_logits = (
                next_logits[top_ids]
                / temperature
            )

            # ------------------------------
            # Stable softmax
            # ------------------------------

            top_logits -= np.max(
                top_logits
            )

            probs = np.exp(
                top_logits
            )

            probs /= np.sum(
                probs
            )

            # ------------------------------
            # Sample from top-k only
            # ------------------------------

            next_token = np.random.choice(
                top_ids,
                p=probs
            )

        # ==========================================
        # Append generated token
        # ==========================================

        token_ids.append(
            int(next_token)
        )

        # ==========================================
        # Clear inference caches
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
    # Decode
    # ==========================================

    return tokenizer.decode(token_ids)