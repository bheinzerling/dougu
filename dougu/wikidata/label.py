from dougu import (
    cached_property,
    )
from dougu.transformer_mixin import TransformerEncoder

from .wikidata_attribute import WikidataAttribute


class WikidataLabel(WikidataAttribute, TransformerEncoder):
    key = 'label'
    args = [
        ('--wikidata-label-max-seq-len', dict(type=int, default=32)),
        ]

    def __init__(self, conf, *args, **kwargs):
        super().__init__(conf, *args, **kwargs)
        self._max_seq_len = self.conf.wikidata_label_max_seq_len

    @property
    def conf_fields(self):
        fields = super().conf_fields
        fields.extend([
            'wikidata_label_max_seq_len',
            'transformer',
            ])
        return fields

    @cached_property
    def raw(self):
        lang = self.wikidata.conf.wikidata_label_lang
        return [self.of(inst, lang) for inst in self.wikidata.raw['train']]

    def of(self, inst, lang):
        return inst.get(self.key, {}).get(lang, inst['id'])

    @cached_property
    def tensor(self):
        return self.encode_texts(self.raw)


class WikidataAliases(WikidataLabel):
    key = 'alias'

    args = WikidataLabel.args + [
        ('--wikidaata-max-aliases', dict(type=int, default=32)),
        ]

    def of(self, inst, lang):
        aliases = inst.get(self.key, {})
        label = [self.wikidata.label.of(inst, lang)]
        return aliases.get(lang, label)

    @cached_property
    def tensor(self):
        raise NotImplementedError('todo')
