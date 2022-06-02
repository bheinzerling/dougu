import torch

from dougu import (
    cached_property,
    file_cached_property,
    )
from dougu.transformer_mixin import WithTransformerEncoder

from .wikidata_attribute import WikidataAttribute


class WikidataDescription(WikidataAttribute, WithTransformerEncoder):
    args = [
        ('--wikidata-desc-max-seq-len', dict(type=int, default=32)),
        ('--wikidata-desc-lang', dict(type=str, default='en')),
        ]

    def __init__(self, conf, *args, **kwargs):
        super().__init__(conf, *args, **kwargs)
        self._max_seq_len = self.conf.wikidata_desc_max_seq_len

    @property
    def conf_fields(self):
        fields = super().conf_fields
        fields.extend([
            'wikidata_desc_max_seq_len',
            'wikidata_desc_lang',
            'transformer',
            ])
        return fields

    def of(self, inst, lang):
        default_desc = 'something'
        return inst.get('description', {}).get(lang, default_desc)

    @cached_property
    def raw(self):
        lang = self.conf.wikidata_desc_lang
        return [self.of(inst, lang) for inst in self.wikidata.raw['train']]

    @file_cached_property
    def tensor(self):
        return self.encode_texts(self.raw)
