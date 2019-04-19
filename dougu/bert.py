from pathlib import Path
import torch

import pytorch_pretrained_bert as _bert

import numpy as np

from dougu import flatten, lines


_device = torch.device("cuda")


class Bert():

    MASK = "[MASK]"
    CLS = "[CLS]"
    SEP = "[SEP]"

    supported_langs = set(lines(
        Path(__file__).parent / "data" / "bert_langs.wiki"))

    def __init__(self, model, model_name, device=None, half_precision=False):
        super().__init__()
        self.model_name = model_name
        self.device = device or _device
        do_lower_case = "uncased" in model_name
        self.tokenizer = _bert.BertTokenizer.from_pretrained(
            self.model_name, do_lower_case=do_lower_case)
        maybe_model_wrapper = model.from_pretrained(model_name).to(
            device=self.device)
        self.maybe_model_wrapper = maybe_model_wrapper
        try:
            self.model = maybe_model_wrapper.bert
        except AttributeError:
            self.model = maybe_model_wrapper
        if half_precision:
            self.model.half()
        self.max_len = \
            self.model.embeddings.position_embeddings.weight.size(0)
        self.dim = self.model.embeddings.position_embeddings.weight.size(1)

    def __call__(self, *args, **kwargs):
        return self.maybe_model_wrapper(*args, **kwargs)

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return self.convert_tokens_to_ids(tokens, pad=pad)

    def mask_mention_and_tokenize_context_to_ids(
            self,
            left_ctx, mention, right_ctx,
            collapse_mask=True,
            pad=True):
        left_ctx_tokenized = self.tokenize(left_ctx)[:-1]  # remove [SEP]
        if collapse_mask:
            masked_mention = [self.MASK]
        else:
            mention_tokenized = self.tokenize(mention)
            masked_mention = [self.MASK] * len(mention_tokenized)
        right_ctx_tokenized = self.tokenize(right_ctx)[1:]  # remove [CLS]
        tokens = left_ctx_tokenized + masked_mention + right_ctx_tokenized
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids]).to(device=self.device)
        assert ids.size(1) < self.max_len
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.

        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.

        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(flatten(subwords)) + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(
            self,
            tokens,
            mask_start_idx=None,
            mask_end_idx=None,
            collapse_mask=True):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.

        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.

        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        if mask_start_idx is not None:
            if mask_end_idx is None:
                mask_end_idx = mask_start_idx + 1
            mask_repeats = 1 if collapse_mask \
                else (mask_end_idx - mask_start_idx)
            tokens = (
                tokens[:mask_start_idx + 1] +
                [self.MASK] * mask_repeats +
                tokens[mask_end_idx:])
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, padding_mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(1, self.max_len).to(subword_ids)
        token_starts[0, token_start_idxs] = 1
        return subword_ids, padding_mask, token_starts

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids]).to(device=self.device)

    @staticmethod
    def Model(model_name, **kwargs):
        return Bert(_bert.BertModel, model_name, **kwargs)

    @staticmethod
    def ForMaskedLM(model_name, **kwargs):
        return Bert(_bert.BertForMaskedLM, model_name, **kwargs)

    @staticmethod
    def ForSequenceClassification(model_name, **kwargs):
        return Bert(
            _bert.BertForSequenceClassification, model_name, **kwargs)

    @staticmethod
    def ForNextSentencePrediction(model_name, **kwargs):
        return Bert(_bert.BertForNextSentencePrediction, model_name, **kwargs)

    @staticmethod
    def ForPreTraining(model_name, **kwargs):
        return Bert(_bert.BertForPreTraining, model_name, **kwargs)

    @staticmethod
    def ForQuestionAnswering(model_name, **kwargs):
        return Bert(_bert.BertForQuestionAnswering, model_name, **kwargs)
