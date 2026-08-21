import re
def build_token_frequencies(raw_data):
    d={} 
    tokens = re.findall(r"\w+|\s+|[^\w\s]", raw_data)
    for token in tokens:
        d[tuple(token)] = d.get(tuple(token), 0) + 1 
    return d
    
def merge_pairs(vocab,d):
    merges=[]
    for _ in range(150):
        # Count pair frequencies
        pair_freq = {}
        for word, freq in d.items():
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
        if not pair_freq:
            break
        # Find best pair
        best_pair = max(pair_freq, key=pair_freq.get)
    
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
    
        # Merge it
        new_d = {}
    
        for word, freq in d.items():
            new_word = []
            i = 0
    
            while i < len(word):
                if (
                    i < len(word)-1 and
                    (word[i], word[i+1]) == best_pair
                ):
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
    
            new_word = tuple(new_word)
            new_d[new_word] = new_d.get(new_word, 0) + freq
    
        d = new_d
    return merges,vocab,d

def build_tokenizer(raw_data, num_merges=150):

    # Initial token frequencies
    d = build_token_frequencies(
        raw_data
    )

    # Initial character vocabulary
    vocab = set()

    for word in d:
        vocab.update(word)

    # BPE training
    merges, vocab, d = merge_pairs(
        vocab,
        d
    )

    # Final vocabulary after merges
    final_vocab = set()

    for word in d:
        final_vocab.update(word)

    final_vocab.add("<unk>")
    vocab = {
            token: idx
            for idx, token
            in enumerate(
                sorted(final_vocab)
            )
        }
    
    # Store BPE merge order
    merge_rank = {
        pair: rank
        for rank, pair
        in enumerate(merges)
    }

    tokenizer = TokenizerV2(
        vocab,
        merge_rank
    )

    return tokenizer, vocab, merge_rank

#tokenizationV2(version2) uses BPE(Byte Pair Encoding) to encode text
#encode uses bpe_encode to encode the text
#then decode follows respectively
class TokenizerV2:
    def __init__(self, vocab, merge_rank):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
        self.merge_rank = merge_rank

    def bpe_encode(self, word):
        word = list(word)

        while len(word) > 1:
            best_pair = None
            best_rank = float("inf")

            # Find the mergeable pair with the smallest rank
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                    best_rank = self.merge_rank[pair]
                    best_pair = pair

            if best_pair is None:
                break

            # Merge every occurrence of that pair
            new_word = []
            i = 0

            while i < len(word):
                if (
                    i < len(word) - 1
                    and (word[i], word[i + 1]) == best_pair
                ):
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

        return word

    def encode(self, text):
        tokens = re.findall(r"\w+|\s+|[^\w\s]", text)

        ids = []

        for token in tokens:
            pieces = self.bpe_encode(token)

            for piece in pieces:
                ids.append(
                    self.str_to_int.get(piece, self.str_to_int["<unk>"])
                )

        return ids

    def decode(self, ids):
        return "".join(self.int_to_str[i] for i in ids)