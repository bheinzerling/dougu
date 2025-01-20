from collections import defaultdict
import re

import torch

from dougu import (
    Configurable,
    cached_property,
    )


class TransformerEncoder(Configurable):
    args = [
        ('--max-seq-len', dict(type=int, default=48)),
        ('--max-new-tokens', dict(type=int, default=64)),
        ('--trf-enc-batch-size', dict(type=int, default=8)),
        ('--trf-enc-device', dict(type=str, default='cuda:0')),
        ('--transformer', dict(type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')),
        ('--trf-include-dec-states', dict(action='store_true')),
        ('--trf-no-generate', dict(action='store_true')),
        ('--trf-rand-init', dict(action='store_true')),
        ('--device-map', dict(type=str, default='balanced_low_0')),
        ('--custom-device-map', dict(action='store_true')),
        ('--show-reconstruction-error-examples', dict(action='store_true')),
        ('--reconstruction-errors', dict(
            type=str,
            choices=[
                'allow',
                'allow_for_lossy_tokenizers',
                'do_not_allow',
                'allow_one_percent',
                'allow_lossy_or_one_percent',
                ],
            default='allow_lossy_or_one_percent')),
        ]
    _max_seq_len = None
    default_device_map = 'balanced_low_0'

    def __init__(self, *args, transformer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = transformer

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'max_seq_len',
            'transformer',
            'trf_rand_init',
            'max_new_tokens',
            ]

    @cached_property
    def tokenizer(self):
        tokenizer = self.load_tokenizer(
            tokenizer_name=self.conf.transformer,
            )
        self.trf_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer

    @staticmethod
    def load_tokenizer(tokenizer_name):
        import os
        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @cached_property
    def trf_config(self):
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(self.conf.transformer)

    @cached_property
    def device_map(self):
        from accelerate import infer_auto_device_map
        trf2device_map = {}
        n_gpus = torch.cuda.device_count()
        if getattr(self.conf, 'custom_device_map', False):
            trf2device_map = {
                ('google/flan-t5-xxl', 4): infer_auto_device_map(
                    self.trf_empty,
                    max_memory={
                        0: '10GiB',
                        1: '13GiB',
                        2: '13GiB',
                        3: '13GiB',
                        },
                    no_split_module_classes=['T5Block'],
                    )
                }
        key = (self.conf.transformer, n_gpus)
        device_map = trf2device_map.get(
            key,
            getattr(self.conf, 'device_map', self.default_device_map)
            )
        if getattr(self.conf, 'custom_device_map', False):
            if hasattr(self, 'log'):
                self.log(f'device map for {key}: {device_map}')
        if device_map == 'balanced_low_0' and n_gpus == 1:
            # 'balanced_low_0' gives a division by zero error with 1 GPU
            device_map = 'auto'
        return device_map

    @cached_property
    def model_cls(self):
        import transformers
        architectures = self.trf_config.architectures
        assert len(architectures) == 1
        architecture = architectures[0]
        architecture = {
            'BloomModel': 'BloomForCausalLM',
            }.get(architecture, architecture)
        return getattr(transformers, architecture)

    @cached_property
    def trf_empty(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            return self.model_cls.from_pretrained(
                self.conf.transformer,
                torch_dtype='auto',
                )

    @cached_property
    def trf(self):
        if self._transformer is not None:
            return self._transformer
        from transformers import AutoModel
        if getattr(self.conf, 'trf_rand_init', False):
            model = AutoModel.from_config(self.trf_config, torch_dtype='auto')
        else:
            if hasattr(self, 'log'):
                self.log('loading model weights: ' + self.conf.transformer)
            model = AutoModel.from_pretrained(
                self.conf.transformer,
                torch_dtype='auto',
                )
        return self.to_gpu(model)

    @property
    def transformer(self):
        return self.trf

    @property
    def max_seq_len(self):
        return self._max_seq_len or self.conf.max_seq_len

    @property
    def bos_offset(self):
        # use ._bos_token instead of .bos_token to avoid warning
        uses_bos = (
            (self.tokenizer._bos_token is not None) and
            (self.tokenizer('test')['input_ids'][0] == self.tokenizer.bos_token_id)
            )
        uses_cls = (self.tokenizer._cls_token is not None)
        return int(uses_bos or uses_cls)

    @property
    def eos_offset(self):
        uses_eos = (
            (self.tokenizer._eos_token is not None) and
            (self.tokenizer('test')['input_ids'][-1] == self.tokenizer.eos_token_id)
            )
        return int(uses_eos)

    @property
    def is_generator(self):
        return (
            not self.conf.trf_no_generate and
            hasattr(self.trf, 'decoder') and
            hasattr(self.trf, 'generate')
            )

    def encode_texts(
            self,
            texts,
            output_hidden_states=True,
            output_fp16=False,
            output_device='cpu',
            hidden_states_layer=None,
            add_word_start_end_indices=False,
            report_reconstruction_errors=True,
            remove_tensor_keys=None,
            ):
        from tqdm import tqdm
        from boltons.iterutils import chunked

        if remove_tensor_keys is None:
            remove_tensor_keys = set()
        elif isinstance(remove_tensor_keys, str):
            remove_tensor_keys = set([remove_tensor_keys])
        else:
            remove_tensor_keys = set(remove_tensor_keys)

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
            chunk_tensors = dict(tok_out)
            if add_word_start_end_indices:
                self._add_word_start_end_indices(
                    tok_out=tok_out, chunk=chunk, chunk_tensors=chunk_tensors)
            with torch.no_grad():
                self.trf.eval()
                self.prepare_model_inputs(tok_out)
                if self.is_generator:
                    max_length = self.max_seq_len + self.conf.max_new_tokens
                    add_kwargs = dict(
                        return_dict_in_generate=True,
                        max_length=max_length,
                        )
                    if self.conf.trf_no_generate:
                        enc_fn = self.trf
                        dec_input_ids = self.tokenizer(
                            "", return_tensors="pt").input_ids
                        if hasattr(self.trf, '_shift_right'):
                            dec_input_ids = self.trf._shift_right(dec_input_ids)
                        add_kwargs['decoder_input_ids'] = dec_input_ids
                    else:
                        enc_fn = self.trf.generate
                else:
                    enc_fn = self.trf
                    add_kwargs = dict()

                trf_out = enc_fn(
                    **tok_out.to(self.conf.trf_enc_device),
                    output_hidden_states=output_hidden_states,
                    **add_kwargs,
                    )

                if self.is_generator:
                    bs, seq_len = trf_out.sequences.shape
                    pad_len = max_length - seq_len
                    if pad_len > 0:
                        pad_id = self.tokenizer.pad_token_id
                        padding = torch.full((bs, pad_len), pad_id)
                        padding = padding.to(trf_out.sequences)
                        padded = torch.cat([trf_out.sequences, padding], dim=1)
                        trf_out.sequences = padded
            if output_hidden_states:
                keys = [
                    'hidden_states',
                    'encoder_hidden_states',
                    ]
                if self.conf.trf_include_dec_states:
                    keys.append('decoder_hidden_states')
                for key in keys:
                    states = trf_out.get(key, None)
                    if states is None:
                        continue
                    if isinstance(states[0], tuple):
                        if len(states) != 1:
                            raise NotImplementedError('TODO: select best decoder_hidden_state from trf.generate() output')
                        states = states[0]
                    device = states[0].device
                    states = [state.to(device=device) for state in states]
                    # stack states so that the shape becomes:
                    # (batch_size, n_layers, seq_len, hidden_size)
                    states = torch.stack(states, dim=1)
                    if hidden_states_layer is not None:
                        idx = hidden_states_layer
                        if idx == -1:
                            idx = states.shape[1] - 1
                        assert 0 <= idx < states.shape[1]
                        states = states[:, idx:idx + 1]
                    trf_out[key] = states

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
                if k in remove_tensor_keys:
                    continue

                if isinstance(v, torch.Tensor):
                    v = v.to(device=output_device)
                    if output_device == 'cpu':
                        if torch.is_floating_point(v):
                            v = v.float()
                    tensors[k].append(v)

        tensors = {k: torch.cat(v) for k, v in tensors.items()}
        if report_reconstruction_errors:
            self.report_reconstruction_errors(texts, tensors)
        return tensors

    def report_reconstruction_errors(
            self, texts, tensors=None, **tokenize_kwargs):
        if tensors is None:
            tensors = self.tokenize(texts, **tokenize_kwargs)
        error_stats, diff_inst = self.reconstruction_error_stats(
            texts, tensors)
        print('tokenizer roundtrip reconstruction error report')
        for k, v in error_stats.items():
            print(f'{k}: {v}')
        if self.conf.show_reconstruction_error_examples:
            print('error examples')
            for t, r in diff_inst[:5]:
                print('original:', t)
                print('reconstr:', r)
                print('---')
        self.maybe_assert_no_tokenizer_reconstruction_errors(error_stats, diff_inst)

    @property
    def lossy_tokenizers(self):
        return {
            'google/flan-t5-xl',
            'google/flan-t5-xxl',
            }

    @property
    def is_lossy_tokenizer(self):
        return self.tokenizer.name_or_path in self.lossy_tokenizers

    def maybe_assert_no_tokenizer_reconstruction_errors(self, error_stats, diff_inst):
        tolerance = 0.0
        match self.conf.reconstruction_errors:
            case 'allow':
                should_check = False
            case 'do_not_allow':
                should_check = True
            case 'allow_for_lossy_tokenizers':
                should_check = not self.is_lossy_tokenizer
            case 'allow_one_percent':
                should_check = True
                tolerance = 0.01
            case 'allow_lossy_or_one_percent':
                should_check = not self.is_lossy_tokenizer
                tolerance = 0.01
        if should_check:
            if tolerance > 0:
                diff_ratio = error_stats['different'] / error_stats['n_inst']
                assert diff_ratio <= tolerance
            else:
                assert error_stats['different'] == 0, diff_inst[:5]

    def reconstruction_error_stats(self, texts, tensors):
        reconstructed_texts = self.tokenizer.batch_decode(
            tensors['input_ids'], skip_special_tokens=True)
        stats = defaultdict(int)
        diff_inst = []
        if self.tokenizer_treats_newlines_as_space:
            texts = [re.sub('[\n ]+', ' ', text ) for text in texts]
        for t, r in zip(texts, reconstructed_texts):
            stats['n_inst'] += 1
            if t == r:
                stats['same'] += 1
            else:
                stats['different'] += 1
                diff_inst.append((t, r))
                if t.startswith(r):
                    stats['truncated'] += 1
        return stats, diff_inst

    @property
    def tokenizer_treats_newlines_as_space(self):
        return self.trf.config._name_or_path in {
            'google/flan-t5-xl',
            'google/flan-t5-xxl',
            }

    @cached_property
    def is_pad_left(self):
        return self.tokenizer.padding_side == 'left'

    def _add_word_start_end_indices(self, *, tok_out, chunk, chunk_tensors):
        subw_lens = tok_out['attention_mask'].sum(dim=1)
        word_start_mask = torch.zeros_like(tok_out['input_ids'])
        word_start_mask[:, self.bos_offset] = 1
        word_ids = torch.zeros_like(tok_out['input_ids']) - 100

        word_start_idxs = torch.zeros_like(tok_out['input_ids'])
        word_end_idxs = torch.zeros_like(tok_out['input_ids'])

        for i, text in enumerate(chunk):
            subw_len = subw_lens[i]
            if self.is_pad_left:
                raise NotImplementedError()
            else:
                word_id_slice = slice(self.bos_offset, subw_len - 1)
                word_start_slice = slice(self.bos_offset + 1, subw_len - 1)
            word_id = torch.tensor(tok_out.word_ids(i)[word_id_slice])
            word_ids[i, word_id_slice] = word_id
            diff = torch.diff(word_id)
            word_start_mask[i, word_start_slice] = diff
            n_words = len(text)
            for word_idx in range(n_words):
                token_span = tok_out.word_to_tokens(i, word_idx)
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

    def decode_input_ids(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def prepare_model_inputs(self, tok_out):
        pass

    def to_gpu(self, model, device_map=None):
        n_devices = torch.cuda.device_count()
        if n_devices == 1:
            model.to(device=self.conf.trf_enc_device)
        else:
            # model.to(device=self.conf.trf_enc_device)
            if device_map is None:
                import math
                from boltons.iterutils import chunked
                n_modules = len(model.encoder.block)
                n_chunks = math.ceil(n_modules / (n_devices - 1))
                device_map = dict(enumerate(
                    chunked(range(n_modules), n_chunks), start=0))
            model.parallelize(device_map)
        return model


class TransformerLM(TransformerEncoder):

    @cached_property
    def trf(self):
        if self._transformer is not None:
            return self._transformer
        model_cls = self.model_cls
        if getattr(self.conf, 'trf_rand_init', False):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.conf.transformer)
            try:
                model = model_cls.from_config(
                    config,
                    device_map=self.device_map,
                    torch_dtype='auto',
                    )
            except ValueError:
                model = model_cls.from_config(
                    config,
                    torch_dtype='auto',
                    )
        else:
            if hasattr(self, 'log'):
                self.log('loading model weights: ' + self.conf.transformer)
            try:
                model = model_cls.from_pretrained(
                    self.conf.transformer,
                    device_map=self.device_map,
                    torch_dtype='auto',
                    )
            except ValueError:
                model = model_cls.from_pretrained(
                    self.conf.transformer,
                    torch_dtype='auto',
                    )
                model = self.to_gpu(model)
        return model

    def prepare_model_inputs(self, tok_out):
        if not hasattr(self.trf, 'prepare_inputs_for_generation'):
            return
        gen_inputs = self.trf.prepare_inputs_for_generation(**tok_out)
        if 'attention_mask' in gen_inputs:
            seq_len = tok_out.attention_mask.size(1)
            gen_attn_mask = gen_inputs['attention_mask'][:, :seq_len]
            assert (gen_attn_mask == tok_out.attention_mask).all()
        for k, v in gen_inputs.items():
            if v is not None and k not in tok_out.data:
                tok_out.data[k] = v

    def tokenize(self, texts, *args, **kwargs):
        return self.tokenizer(
            texts,
            *args,
            return_tensors='pt',
            **kwargs,
            ).to(self.trf.device)

    def generate(
            self,
            prompt,
            *args,
            temperature=1e-8,
            return_new_tokens_only=True,
            max_new_tokens=None,
            **kwargs,
            ):
        tok_out = self.tokenize(prompt)
        if max_new_tokens is None:
            max_new_tokens = self.conf.max_new_tokens
        trf_out = self.trf.generate(
            tok_out.input_ids,
            *args,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
            )
        text = self.tokenizer.batch_decode(
            trf_out,
            skip_special_tokens=True,
            )
        assert len(text) == 1
        text = text[0]
        if return_new_tokens_only and text.startswith(prompt):
            text = text[len(prompt):]
        return text

    def encode(self, texts, labels=None, **kwargs):
        tok_out = self.tokenize(texts)
        if labels is not None:
            labels = self.tokenize(labels).input_ids
        return self.trf(**tok_out, labels=labels, **kwargs)

    def decode(self, trf_out):
        output_ids = trf_out.logits.argmax(dim=-1)
        return self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            )
