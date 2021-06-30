from transformers import AutoTokenizer

from dougu import cached_property


class WithTokenizer():

    max_seq_len = 512

    @cached_property
    def tokenizer(self):
        tok = AutoTokenizer.from_pretrained(
            self.conf.transformer,
            add_prefix_space=True,
            )
        if hasattr(self.conf, 'max_seq_len'):
            self.max_seq_len = self.conf.max_seq_len
        return tok

    def encode_texts(self, texts):
        from collections import defaultdict
        import torch
        from boltons.iterutils import chunked
        from tqdm import tqdm

        if isinstance(texts, str):
            texts = [texts]

        is_split_into_words = not isinstance(texts[0], str)

        tensors = defaultdict(list)
        disable_tqdm = len(texts) < 100
        for chunk in tqdm(chunked(texts, 5000), disable=disable_tqdm):
            tokenizer_out = self.tokenizer(
                chunk,
                max_length=self.max_seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                is_split_into_words=is_split_into_words,
                )
            chunk_tensors = dict(tokenizer_out)
            if is_split_into_words:
                subw_lens = tokenizer_out['attention_mask'].sum(dim=1)
                word_start_mask = torch.zeros_like(tokenizer_out['input_ids'])
                word_start_mask[:, 1] = 1
                word_ids = torch.zeros_like(tokenizer_out['input_ids']) - 100

                word_start_idxs = torch.zeros_like(tokenizer_out['input_ids'])
                word_end_idxs = torch.zeros_like(tokenizer_out['input_ids'])

                for i, text in enumerate(chunk):
                    subw_len = subw_lens[i]
                    word_id = torch.tensor(
                        tokenizer_out.word_ids(i)[1:subw_len - 1])
                    word_ids[i, 1:subw_len - 1] = word_id
                    diff = torch.diff(word_id)
                    word_start_mask[i, 2:subw_len - 1] = diff
                    n_words = len(text)
                    for word_idx in range(n_words):
                        token_span = tokenizer_out.word_to_tokens(i, word_idx)
                        if token_span is None:
                            # instance was truncated
                            break
                        word_start_idxs[i, word_idx] = token_span.start
                        word_end_idxs[i, word_idx] = token_span.end

                chunk_tensors.update({
                    'word_start_mask': word_start_mask,
                    'word_start_idxs': word_start_idxs,
                    'word_end_idxs': word_end_idxs,
                    })
            for k, v in chunk_tensors.items():
                tensors[k].append(v)

        return {k: torch.cat(v) for k, v in tensors.items()}
