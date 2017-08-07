from datetime import datetime
from collections import Counter


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_emb_dim(emb):
    return emb[emb.index2word[0]].shape[0]


def unk_emb_stats(sentences, emb):
    """Compute some statistics about unknown tokens in sentences
    such as "how many sentences contain an unknown token?".
    emb can be gensim KeyedVectors or any other object implementing
    __contains__
    """
    stats = {
        "sents": 0,
        "tokens": 0,
        "unk_tokens": 0,
        "unk_types": 0,
        "unk_tokens_lower": 0,
        "unk_types_lower": 0,
        "sents_with_unk_token": 0,
        "sents_with_unk_token_lower": 0}

    all_types = set()

    for sent in sentences:
        stats["sents"] += 1
        any_unk_token = False
        any_unk_token_lower = False
        types = Counter(sent)
        for ty, freq in types.items():
            all_types.add(ty)
            stats["tokens"] += freq
            unk = ty not in emb
            if unk:
                any_unk_token = True
                stats["unk_types"] += 1
                stats["unk_tokens"] += freq
            if unk and ty.lower() not in emb:
                any_unk_token_lower = True
                stats["unk_types_lower"] += 1
                stats["unk_tokens_lower"] += freq
        if any_unk_token:
            stats["sents_with_unk_token"] += 1
        if any_unk_token_lower:
            stats["sents_with_unk_token_lower"] += 1
    stats["types"] = len(all_types)

    return stats


def to_word_indexes(tokens, keyedvectors, unk=None):
    if unk is None:
        return [keyedvectors.vocab[token].index for token in tokens] 
    unk = keyedvectors.vocab[unk]
    return [keyedvectors.vocab.get(token, unk).index for token in tokens]
