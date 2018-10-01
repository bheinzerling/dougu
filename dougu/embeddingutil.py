import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_word2vec_file(
        word2vec_file,
        weights_file=None, normalize=False, add_unk=False, unk="<unk>",
        add_pad=False, pad="<pad>"):
    """Load a word2vec file in either text or bin format, optionally
    supplying custom embedding weights and normalizing embeddings."""
    from gensim.models import KeyedVectors
    word2vec_file = str(word2vec_file)
    binary = word2vec_file.endswith(".bin")
    log.info("loading %s", word2vec_file)
    vecs = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
    if add_unk:
        if unk not in vecs:
            add_unk_embedding(vecs)
        else:
            pass
            # raise ValueError("Attempted to add <unk>, but already present")
    if add_pad:
        if pad not in vecs:
            add_embeddings(vecs, pad)
        else:
            raise ValueError("Attempted to add <pad>, but already present")
    if weights_file:
        import torch
        weights = torch.load(weights_file)
        vecs.syn0 = weights.cpu().float().numpy()
    if normalize:
        log.info("normalizing %s", word2vec_file)
        vecs.init_sims(replace=True)
    return vecs


def get_emb_dim(emb):
    return emb[emb.index2word[0]].shape[0]


def unk_emb_stats(sentences, emb):
    """Compute some statistics about unknown tokens in sentences
    such as "how many sentences contain an unknown token?".
    emb can be gensim KeyedVectors or any other object implementing
    __contains__
    """
    from collections import Counter

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


def to_word_indexes(tokens, keyed_vectors, unk=None, fallback_transform=None):
    """Look up embedding indexes for tokens."""
    if fallback_transform:
        assert unk
        unk = keyed_vectors.vocab[unk]
        return [
            keyed_vectors.vocab.get(
                token,
                keyed_vectors.vocab.get(fallback_transform(token), unk)).index
            for token in tokens]
    if unk is None:
        return [keyed_vectors.vocab[token].index for token in tokens]
    unk = keyed_vectors.vocab[unk]
    return [keyed_vectors.vocab.get(token, unk).index for token in tokens]


def add_embeddings(keyed_vectors, *words, init=None):
    import numpy as np
    from gensim.models.keyedvectors import Vocab
    if init is None:
        init = np.zeros
    syn0 = keyed_vectors.syn0
    for word in words:
        keyed_vectors.vocab[word] = Vocab(count=0, index=syn0.shape[0])
        keyed_vectors.syn0 = np.concatenate([syn0, init((1, syn0.shape[1]))])
        keyed_vectors.index2word.append(word)
    return syn0.shape[0]


def add_unk_embedding(keyed_vectors, unk_str="<unk>", init=None):
    """Add a vocab entry and embedding for unknown words to keyed_vectors."""
    add_embeddings(keyed_vectors, unk_str, init)


def mu_postproc(V, D=1):
    """algorithm 1, https://arxiv.org/pdf/1702.01417.pdf"""
    import numpy as np
    from sklearn.decomposition import PCA

    assert D >= 1
    V_ = V - V.mean(0)
    pca = PCA()
    pca.fit(V_)
    print("var ratios:", pca.explained_variance_ratio_)
    print(pca.components_.shape)
    s = np.zeros(V.shape)
    for i in range(D):
        s += np.repeat(
            (pca.components_[i] * V_).sum(1)[:, None], V.shape[1], 1) * \
                    pca.components_[i]
    return V_ - s


def plot_emb(emb, n=1000):
    import numpy as np
    from .plot import simple_imshow
    mask = np.random.choice(np.arange(emb.syn0.shape[0]), n)
    simple_imshow(emb.syn0[mask].transpose())
