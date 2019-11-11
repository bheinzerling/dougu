from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModel

import numpy as np

from dougu import flatten, lines, get_logger


_device = torch.device("cuda:0")


class Transformer():

    MASK = "[MASK]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    BOS = "<s>"
    EOS = "</s>"
    BEGIN_MENTION = '[unused0]'
    END_MENTION = '[unused1]'
    nspecial_symbols_segment1 = 2  # [CLS] sent1... [SEP]
    nspecial_symbols_segment2 = 1  # sent2... [SEP]
    add_tokens_key = 'additional_special_tokens'
    supported_langs = set(lines(
        Path(__file__).parent / "data" / "bert_langs.wiki"))

    def __init__(self, model_name, device=None, max_len=None):
        super().__init__()
        self.model_name = model_name
        self.device = device or _device
        self.log = get_logger()
        do_lower_case = "uncased" in model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, do_lower_case=do_lower_case)
        self.begin_mention_idx = self.tokenizer.convert_tokens_to_ids(
            self.BEGIN_MENTION)
        additional_special_tokens = [self.BEGIN_MENTION, self.END_MENTION]
        self.tokenizer.add_special_tokens({
            self.add_tokens_key: additional_special_tokens})
        self.model = AutoModel.from_pretrained(model_name)
        device_count = torch.cuda.device_count()
        self.log.info(f'device count: {device_count}')
        if device_count > 1:
            # device_ids = list(range(device_count))
            self.model = torch.nn.DataParallel(self.model)
            self.module = self.model.module
            # self.model.to(device='cuda')
        else:
            self.module = self.model
            self.model.to(device=self.device)
        self.max_len = max_len or self.tokenizer.max_len
        self.dim = self.module.embeddings.position_embeddings.weight.size(1)
        if self.model_name.startswith('roberta'):
            self.add_special_symbols = self.add_special_symbols_roberta
        else:
            self.add_special_symbols = self.add_special_symbols_bert
        self.BEGIN_MENTION_IDX = self.tokenizer.convert_tokens_to_ids(
            self.BEGIN_MENTION)
        self.END_MENTION_IDX = self.tokenizer.convert_tokens_to_ids(
            self.END_MENTION)

    def update_special_tokens(self, additional_special_tokens):
        current = self.tokenizer.special_tokens_map[self.add_tokens_key]
        self.tokenizer.add_special_tokens({
            self.add_tokens_key: current + additional_special_tokens})

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def tokenize(self, text, masked_idxs=None):
        if isinstance(text, str):
            tokenized_text = self.tokenizer.tokenize(text)
            if masked_idxs is not None:
                for idx in masked_idxs:
                    tokenized_text[idx] = self.MASK
            tokenized = self.add_special_symbols(tokenized_text)
            return tokenized
        return list(map(self.tokenize, text))

    def add_special_symbols_bert(self, tokenized_text):
        return [self.CLS] + tokenized_text + [self.SEP]

    def add_special_symbols_roberta(self, tokenized_text):
        return [self.BOS] + tokenized_text + [self.EOS]

    def tokenize_sentence_pair(self, sent1, sent2):
        tokenized_sent1 = self.tokenizer.tokenize(sent1)
        tokenized_sent2 = self.tokenizer.tokenize(sent2)
        return self.add_special_symbols_sent_pair(
            tokenized_sent1, tokenized_sent2)

    def add_special_symbols_sent_pair(
            self, tokenized_sent1, tokenized_sent2):
        return (
            [self.CLS] + tokenized_sent1 + [self.SEP] +
            tokenized_sent2 + [self.SEP])

    def tokenize_to_ids(
            self, text, masked_idxs=None, pad=True, clip_long_seq=False):
        tokens = self.tokenize(text, masked_idxs)
        return self.convert_tokens_to_ids(
            tokens, pad=pad, clip_long_seq=False)

    def tokenize_sentence_pair_to_ids(self, sent1, sent2):
        tokenized_sent1 = self.tokenizer.tokenize(sent1)
        segment1_len = len(tokenized_sent1) + self.nspecial_symbols_segment1
        tokenized_sent2 = self.tokenizer.tokenize(sent2)
        segment2_len = len(tokenized_sent2) + self.nspecial_symbols_segment2
        tokenized_sents = self.add_special_symbols(
            tokenized_sent1, tokenized_sent2)
        padded_ids, padding_mask = self.convert_tokens_to_ids(tokenized_sents)
        segment_ids = self.segment_ids(segment1_len, segment2_len)
        return padded_ids, padding_mask, segment_ids

    def mask_mention_and_tokenize_context(
            self, collapse_mask, *, left_ctx, mention, right_ctx, **kwargs):
        left_ctx_tokenized = self.tokenize(left_ctx)[:-1]  # remove [SEP]
        if collapse_mask:
            masked_mention = [self.MASK]
        else:
            mention_tokenized = self.tokenize(mention)
            masked_mention = [self.MASK] * len(mention_tokenized)
        right_ctx_tokenized = self.tokenize(right_ctx)[1:]  # remove [CLS]
        tokens = left_ctx_tokenized + masked_mention + right_ctx_tokenized
        return tokens

    def mask_mention_and_tokenize_context_to_ids(
            self,
            left_ctx, mention, right_ctx,
            collapse_mask=True,
            pad=True):
        tokens = self.mask_mention_and_tokenize_context(
            collapse_mask=collapse_mask,
            left_ctx=left_ctx,
            mention=mention,
            right_ctx=right_ctx)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def mask_mentions_and_tokenize_contexts_to_ids(
            self,
            mentions_and_contexts,
            collapse_mask=True):
        tokens = [
            self.mask_mention_and_tokenize_context(
                collapse_mask=collapse_mask, **ment_ctx)
            for ment_ctx in mentions_and_contexts]
        return tokens, self.convert_tokens_to_ids(tokens)

    def convert_tokens_to_ids(self, tokens, pad=True, clip_long_seq=False):
        if isinstance(tokens[0], list):
            token_idss = map(self.tokenizer.convert_tokens_to_ids, tokens)
            padded_ids = torch.zeros(
                (len(tokens,), self.max_len), dtype=torch.long)
            for row_idx, token_ids in enumerate(token_idss):
                token_ids = torch.tensor(token_ids)
                padded_ids[row_idx, :len(token_ids)] = token_ids
            padded_ids = padded_ids.to(device=self.device)
            mask = padded_ids > 0
            return padded_ids, mask
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids]).to(device=self.device)
        if clip_long_seq:
            ids = ids[:, :self.max_len]
        else:
            assert ids.size(1) < self.max_len
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def subword_tokenize(
            self,
            tokens,
            mask_start_idx=None,
            mask_end_idx=None,
            add_mask_start_end_markers=False,
            collapse_mask=True,
            apply_mask=True,
            clip_long_seq=False):
        """Segment each token into subwords while keeping track of
        token boundaries.

        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.

        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the required special symbols.
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        if mask_start_idx is not None:
            try:
                mask_starts = list(iter(mask_start_idx))
            except TypeError:
                mask_starts = [mask_start_idx]
            if mask_end_idx is None:
                assert len(mask_starts) == 1
                mask_ends = [mask_starts[0] + 1]
            else:
                try:
                    mask_ends = list(iter(mask_end_idx))
                except TypeError:
                    mask_ends = [mask_end_idx]

            mask_start_ends = list(reversed(list(zip(mask_starts, mask_ends))))
            if apply_mask:
                for mask_start, mask_end in mask_start_ends:
                    mask_len = 1 if collapse_mask else (mask_end - mask_start)
                    tokens = (
                        tokens[:mask_start] +
                        [self.MASK] * mask_len +
                        tokens[mask_end:])
            if add_mask_start_end_markers:
                for mask_start, mask_end in mask_start_ends:
                    if apply_mask:
                        mask_len = 1 if collapse_mask else (
                            mask_end - mask_start)
                        mention = [self.MASK] * mask_len
                    else:
                        mention = tokens[mask_start:mask_end]
                    tokens = (
                        tokens[:mask_start] +
                        [self.BEGIN_MENTION] +
                        mention +
                        [self.END_MENTION] +
                        tokens[mask_end:])
                # account for inserted mention markers
                new_mask_starts = [
                    i for i, t in enumerate(tokens)
                    if t == self.BEGIN_MENTION]
                new_mask_ends = [
                    i + 1 for i, t in enumerate(tokens)
                    if t == self.END_MENTION]
                mask_start_ends = list(reversed(list(zip(
                    new_mask_starts, new_mask_ends))))
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = self.add_special_symbols(list(flatten(subwords)))
        # + 1: assumes one special symbol is prepended to the input sequence
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        if mask_start_idx is not None:
            return subwords, token_start_idxs, mask_start_ends
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(
            self,
            tokens,
            mask_start_idx=None,
            mask_end_idx=None,
            add_mask_start_end_markers=False,
            collapse_mask=True,
            apply_mask=True,
            return_mask_mask=False,
            return_mask_start_end=False):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.

        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.

        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the required
                special symbols.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs, mask_start_ends = self.subword_tokenize(
            tokens,
            mask_start_idx=mask_start_idx,
            mask_end_idx=mask_end_idx,
            add_mask_start_end_markers=add_mask_start_end_markers,
            collapse_mask=collapse_mask,
            apply_mask=apply_mask)
        subword_ids, padding_mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(1, self.max_len).to(subword_ids)
        token_starts[0, token_start_idxs] = 1
        if return_mask_mask:
            mask_mask = torch.zeros(1, self.max_len).to(subword_ids)
            for mask_start, mask_end in mask_start_ends:
                token_mask_idxs = list(range(mask_start, mask_end))
                subw_mask_idxs = token_start_idxs[token_mask_idxs]
                mask_mask[0, subw_mask_idxs] = 1
            if return_mask_start_end:
                mask_start_end = torch.zeros(1, self.max_len).to(subword_ids)
                # this only works if there are fewer than seq_len // 2 masks
                for i, (mask_start, mask_end) in enumerate(mask_start_ends):
                    token_mask_idxs = list(range(mask_start, mask_end))
                    subw_mask_idxs = token_start_idxs[token_mask_idxs]
                    mask_start_end[0, 2*i] = int(subw_mask_idxs[0])
                    mask_start_end[0, 2*i+1] = int(subw_mask_idxs[-1])
                return (
                    subword_ids, padding_mask, token_starts,
                    mask_mask, mask_start_end)
            else:
                return subword_ids, padding_mask, token_starts, mask_mask
        return subword_ids, padding_mask, token_starts

    def segment_ids(self, segment1_len, segment2_len, pad=True):
        npad = self.max_len - segment1_len - segment2_len
        ids = [0] * segment1_len + [1] * segment2_len + [0] * npad
        assert len(ids) == self.max_len
        return torch.tensor([ids]).to(device=self.device)
