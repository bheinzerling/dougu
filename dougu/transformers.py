from .decorators import cached_property


class T:
    def __init__(self, model_name, **from_pretrained_kwargs):
        self.model_name = model_name
        self.from_pretrained_kwargs = from_pretrained_kwargs

    @cached_property
    def tokenizer(self):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def config(self):
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(
            self.model_name,
            **self.from_pretrained_kwargs,
            )

    @cached_property
    def trf(self):
        import transformers
        architectures = self.config.architectures
        assert len(architectures) == 1
        architecture = architectures[0]
        model_cls = getattr(transformers, architecture)
        return model_cls.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype='auto',
            **self.from_pretrained_kwargs,
            )

    def tokenize(self, texts, *args, **kwargs):
        return self.tokenizer(
            texts,
            *args,
            return_tensors='pt',
            **kwargs,
            ).to(self.trf.device)

    def __call__(self, texts, *args, **kwargs):
        trf_out = self.encode(texts, *args, **kwargs)
        return self.decode(trf_out)

    def generate(self, text, *args, **kwargs):
        tok_out = self.tokenize(text)
        trf_out = self.trf.generate(
            *args,
            **tok_out,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
            )
        return self.tokenizer.batch_decode(
            trf_out,
            skip_special_tokens=True,
            )

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
