from dougu import (
    Configurable,
    WithLog,
    )


class WikidataAttribute(Configurable, WithLog):
    def __init__(self, conf, wikidata, *args, **kwargs):
        super().__init__(conf, *args, **kwargs)
        self.wikidata = wikidata

    @property
    def conf_fields(self):
        fields = [
            'wikidata_top_n',
            ]
        return fields

    @property
    def entity_ids(self):
        return self.wikidata.entity_ids

    @property
    def n_entities(self):
        return self.wikidata.n_entities

    @property
    def raw(self):
        raise NotImplementedError()

    @property
    def tensor(self):
        raise NotImplementedError()

    def __getitem__(self, wikidata_id):
        if isinstance(wikidata_id, str):
            idx = self.wikidata.entity_id_enc.transform(wikidata_id).item()
            return self.raw[idx]
        return [
            self.raw[idx]
            for idx in self.wikidata.entity_id_enc.transform(wikidata_id)
            ]
