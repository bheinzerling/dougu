from pathlib import Path
from collections import Counter

import numpy as np

import torch

from dougu import (
    lines,
    dict_load,
    cached_property,
    file_cached_property,
    )
from dougu.codecs import LabelEncoder

from .wikidata_attribute import WikidataAttribute


class WikidataNumericAttributes(WikidataAttribute):
    year_unit = 'Q1092296'
    degree_unit = 'Q28390'
    none_label = ''

    args = [
        ('--wikidata-num-attributes-fname',
            dict(type=Path, default='loc_quant_time.pred.most_freq_100')),
        ('--wikidata-units-fname',
            dict(type=Path, default='loc_quant_time.unit')),
        ('--wikidata-unit-labels-fname',
            dict(type=Path, default='loc_quant_time.unit.label_en')),
        ]

    @cached_property
    def attr_names(self):
        return list(map(
            self.property_id2property_name.get,
            self.pred_enc.labels.tolist()))

    @property
    def allowed_attrs_file(self):
        return self.wikidata.data_dir / self.conf.wikidata_num_attributes_fname

    @property
    def allowed_units_file(self):
        return self.wikidata.data_dir / self.conf.wikidata_units_fname

    @cached_property
    def allowed_attrs(self):
        return set(lines(self.allowed_attrs_file))

    @cached_property
    def pred_enc(self):
        return LabelEncoder.from_file(self.allowed_attrs_file)

    @cached_property
    def n_attrs(self):
        return len(self.allowed_attrs)

    @cached_property
    def unit_id2label(self):
        fname = self.conf.wikidata_unit_labels_fname
        unit_id2label = dict_load(self.data_dir / fname, splitter='\t')
        unit_id2label['1'] = '1'
        for unit in self.units:
            if unit not in unit_id2label:
                unit_id2label[unit] = unit
        return unit_id2label

    @cached_property
    def units(self):
        return set(self.unit_enc.labels)

    @cached_property
    def n_units(self):
        return len(self.units)

    @cached_property
    def raw(self):
        def transform_time(wikidata_time):
            t = wikidata_time
            t = int(t[0] + t[1:].split("-")[0])
            return min(2100, max(-1000, t))

        def filter_quantities(numvals, units):
            try:
                numvals, units = zip(*[
                    (v, u) for v, u in zip(numvals, units) if u in self.units])
            except ValueError:
                return [None], [None]
            most_freq_unit = Counter(units).most_common(1)[0][0]
            numvals, units = zip(*[
                (v, u) for v, u in zip(numvals, units) if u == most_freq_unit])
            return numvals, units

        def aggregate_quantities(quantities, aggregate_fn=np.median):
            numvals, units = zip(*quantities)
            unique_units = set(units)
            if len(unique_units) != 1:
                numvals, units = filter_quantities(numvals, units)
                unique_units = set(units)
                assert len(set(units)) == 1
            unit = next(iter(unique_units))
            if unit is None:
                numval = None
            else:
                numval = aggregate_fn(list(map(float, numvals)))
            return numval, unit

        def get_numattrs(inst):
            # in variable names:
            # p = predicate, q = quantity, v = numeric value, u = unit
            # a quantity is a pair of a numeric value and a unit
            qdict = inst.get('quantity', {})
            if 'time' in inst:
                tdict = inst['time']
                titems = [
                    [p, [[transform_time(v), self.year_unit]]]
                    for p, v in tdict.items()]
                qdict.update(titems)
            if 'geocoordinate' in inst:
                if 'P625' in inst['geocoordinate']:
                    p625 = inst['geocoordinate']['P625']
                    gitems = [
                        ['P625.long', [[p625['longitude'], self.degree_unit]]],
                        ['P625.lat', [[p625['latitude'], self.degree_unit]]]]
                    qdict.update(gitems)

            attrs = list(qdict.keys() & self.allowed_attrs)
            if not attrs:
                return [], []
            quantitiess = list(map(qdict.__getitem__, attrs))
            numvals_units = list(map(aggregate_quantities, quantitiess))
            attrs, numvals_units = zip(*[
                [p, vu] for p, vu in zip(attrs, numvals_units)
                if vu[1] is not None])
            return attrs, numvals_units

        return [get_numattrs(inst) for inst in self.wikidata.raw['train']]

    @cached_property
    def sparse_tensors(self):
        import scipy
        assert self.year_unit in self.units
        assert self.degree_unit in self.units
        sparse_vu_raw = [
            (row_idx, col_idx, v, u)
            for row_idx, pvu in enumerate(self.raw)
            for col_idx, (v, u) in [
                (self.pred_enc.transform(p).item(), vu)
                for p, vu in zip(*pvu)
                if p]]
        row_idxs, col_idxs, v_raw, u_raw = zip(*sparse_vu_raw)

        max_val = 2.0**64  # a few values are too large for FP32, clip those
        v_raw = np.array(v_raw).clip(-max_val, max_val)
        u_enc = self.unit_enc.transform(u_raw)

        idxs = (row_idxs, col_idxs)
        shape = (self.n_entities, self.n_attrs)
        v_raw_csc = scipy.sparse.csc_matrix((v_raw, idxs), shape=shape)
        u_csc = scipy.sparse.csc_matrix((u_enc, idxs), shape=shape)
        return {'v_raw_csc': v_raw_csc, 'u_csc': u_csc}

    @file_cached_property(fname_tpl='numval_encs.{conf_str}.pkl')
    def numval_encs(self):
        v_raw_csc = self.sparse_tensors['v_raw_csc']
        from sklearn.preprocessing import QuantileTransformer
        return [
            QuantileTransformer(
                n_quantiles=10000, ignore_implicit_zeros=True
                ).fit(v_raw_csc[:, col_idx])
            for col_idx in range(v_raw_csc.shape[1])]

    @file_cached_property(fname_tpl='unit_enc.{conf_str}.pkl')
    def unit_enc(self):
        unit_enc = LabelEncoder.from_file(
            self.allowed_units_file,
            additional_labels=[self.none_label])
        assert unit_enc.transform(self.none_label) == 0
        return unit_enc

    def sparse_tensors_to_dense(self, sparse_tensors):
        import scipy
        v_raw_csc = sparse_tensors['v_raw_csc']
        u_csc = sparse_tensors['u_csc']
        v_cols = [
            self.numval_encs[col_idx].transform(v_raw_csc[:, col_idx])
            for col_idx in range(v_raw_csc.shape[1])]
        # sparse tensors have the default value 0. To distinguish this from
        # zeros in q , shift q by 1, then undo the shift afterwards,
        # resulting in missing values being represented by -1
        for v_col in v_cols:
            v_col.data += 1  # 'cannot add nonzero scalars to sparse matrices'
        v_csc = scipy.sparse.hstack(v_cols)
        v = torch.tensor(v_csc.toarray() - 1).to(dtype=torch.float)
        u = torch.tensor(u_csc.toarray()).long()
        entity_idx = torch.arange(len(v))
        return {'v': v, 'u': u, 'entity_idx': entity_idx}

    @cached_property
    def tensor(self):
        return self.sparse_tensors_to_dense(self.sparse_tensors)

    def decode(self, v, u, existing_mask=None, only_existing_values=True):
        assert len(v) == len(self.numval_encs)
        v = v.cpu()
        u = u.cpu()
        numvals = [
            numval_enc.inverse_transform(w.reshape(-1, 1))[0, 0]
            for numval_enc, w in zip(self.numval_encs, v)]
        unit_ids = self.unit_enc.inverse_transform(u)

        if only_existing_values:
            if existing_mask is None:
                existing_mask = v != -1
            if not existing_mask.any():
                return [], existing_mask
            attr_name, numvals, unit_ids = zip(*[
                (attr_name, numval, unit_id) for e, attr_name, numval, unit_id in zip(
                    existing_mask, self.attr_names, numvals, unit_ids)
                if e])

        unit_names = list(map(self.unit_id2name.get, unit_ids))
        return list(zip(attr_name, numvals, unit_names)), existing_mask
