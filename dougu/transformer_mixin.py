from collections import defaultdict

import torch

from dougu import (
    Configurable,
    cached_property,
    )


class WithTransformerEncoder(Configurable):
    args = [
        ('--max-seq-len', dict(type=int, default=64)),
        ('--trf-enc-batch-size', dict(type=int, default=32)),
        ('--trf-enc-device', dict(type=str, default='cuda:0')),
        ('--transformer', dict(type=str, default='roberta-base')),
        ]
    _max_seq_len = None

    def __init__(self, *args, transformer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = None

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'max_seq_len',
            'transformer',
            ]

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            self.conf.transformer,
            # add_prefix_space=True,
            )

    @cached_property
    def trf(self):
        if self._transformer is not None:
            return self._transformer
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            self.conf.transformer,
            ).to(device=self.conf.trf_enc_device)

    @property
    def max_seq_len(self):
        return self._max_seq_len or self.conf.max_seq_len

    @property
    def bos_offset(self):
        # use ._bos_token instead of .bos_token to avoid warning
        uses_bos = self.tokenizer._bos_token is None
        uses_cls = self.tokenizer._cls_token is None
        return int(uses_bos) + int(uses_cls)

    def encode_texts(
            self,
            texts,
            output_hidden_states=True,
            output_fp16=False,
            output_device='cpu',
            ):
        from tqdm import tqdm
        from boltons.iterutils import chunked

        if isinstance(texts, str):
            texts = [texts]

        tensors = defaultdict(list)
        disable_tqdm = len(texts) < 100
        batches = chunked(texts, self.conf.trf_enc_batch_size)
        for chunk in tqdm(batches, disable=disable_tqdm):
            tok_out = self.tokenizer(
                chunk,
                max_length=self.max_seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                )
            subw_lens = tok_out['attention_mask'].sum(dim=1)
            word_start_mask = torch.zeros_like(tok_out['input_ids'])
            word_start_mask[:, self.bos_offset] = 1
            word_ids = torch.zeros_like(tok_out['input_ids']) - 100

            word_start_idxs = torch.zeros_like(tok_out['input_ids'])
            word_end_idxs = torch.zeros_like(tok_out['input_ids'])

            for i, text in enumerate(chunk):
                subw_len = subw_lens[i]
                word_id = torch.tensor(
                    tok_out.word_ids(i)[self.bos_offset:subw_len - 1])
                word_ids[i, self.bos_offset:subw_len - 1] = word_id
                diff = torch.diff(word_id)
                word_start_mask[i, self.bos_offset + 1:subw_len - 1] = diff
                n_words = len(text)
                for word_idx in range(n_words):
                    token_span = tok_out.word_to_tokens(i, word_idx)
                    if token_span is None:
                        # instance was truncated
                        break
                    word_start_idxs[i, word_idx] = token_span.start
                    word_end_idxs[i, word_idx] = token_span.end

            chunk_tensors = dict(tok_out)
            chunk_tensors.update({
                'word_start_mask': word_start_mask,
                'word_start_idxs': word_start_idxs,
                'word_end_idxs': word_end_idxs,
                })

            with torch.no_grad():
                self.trf.eval()
                self.prepare_model_inputs(tok_out)
                trf_out = self.trf(
                    **tok_out.to(self.conf.trf_enc_device),
                    output_hidden_states=output_hidden_states,
                    )
            if output_hidden_states:
                keys = [
                    'hidden_states',
                    'encoder_hidden_states',
                    'decoder_hidden_states',
                    ]
                for key in keys:
                    try:
                        trf_out[key] = torch.stack(trf_out[key], dim=1)
                    except KeyError:
                        pass
            chunk_tensors.update(dict(trf_out))
            if output_fp16:
                def to_half(obj):
                    try:
                        if torch.is_floating_point(obj):
                            return obj.half()
                    except TypeError:
                        pass
                    return obj

                chunk_tensors = {
                    k: to_half(v) for k, v in chunk_tensors.items()}

            for k, v in chunk_tensors.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device=output_device)
                    tensors[k].append(v)

        return {k: torch.cat(v) for k, v in tensors.items()}

    def prepare_model_inputs(self, tok_out):
        pass


class WithTransformerLM(WithTransformerEncoder):
    @cached_property
    def trf(self):
        if self._transformer is not None:
            return self._transformer
        from transformers import (
            AutoModelWithLMHead,
            AutoModelForMaskedLM,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            )
        exception = None
        for automodel_cls in [
                AutoModelWithLMHead,
                AutoModelForMaskedLM,
                AutoModelForSeq2SeqLM,
                AutoModelForCausalLM,
                ]:
            try:
                return automodel_cls.from_pretrained(
                    self.conf.transformer,
                    ).to(device=self.conf.trf_enc_device)
            except ValueError as e:
                exception = e
        else:
            raise exception

    def prepare_model_inputs(self, tok_out):
        gen_inputs = self.trf.prepare_inputs_for_generation(**tok_out)
        if 'attention_mask' in gen_inputs:
            seq_len = tok_out.attention_mask.size(1)
            gen_attn_mask = gen_inputs['attention_mask'][:, :seq_len]
            assert (gen_attn_mask == tok_out.attention_mask).all()
        for k, v in gen_inputs.items():
            if v is not None and k not in tok_out.data:
                tok_out.data[k] = v
