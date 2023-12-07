from pathlib import Path
from collections import Counter

import numpy as np

import pandas as pd

import torch

from dougu import (
    lines,
    dict_load,
    cached_property,
    file_cached_property,
    global_cached_property,
    take_singleton,
    )
from dougu.codecs import LabelEncoder

from .wikidata_attribute import WikidataAttribute


def precision_and_type_for_pred_and_unit(pred_id, unit_id):
    date_of_birth = 'P569'
    date_of_death = 'P570'
    inception = 'P571'
    dissolved_date = 'P576'
    publication_date = 'P577'
    point_in_time = 'P585'
    latitude = 'P625.lat'
    longitude = 'P625.long'
    population = 'P1082'
    number_of_households = 'P1538'
    work_period_start = 'P2031'
    elevation = 'P2044'
    area = 'P2046'
    duration = 'P2047'

    annum = 'Q1092296'
    metre = 'Q11573'
    square_km = 'Q712226'
    minute = 'Q7727'
    degree = 'Q28390'
    no_unit = '1'

    pred_id_and_unit_id2precision_and_type = {
        (work_period_start, annum): (0, int),
        (duration, minute): (0, int),
        (date_of_birth, annum): (0, int),
        (date_of_death, annum): (0, int),
        (inception, annum): (0, int),
        (dissolved_date, annum): (0, int),
        (publication_date, annum): (0, int),
        (point_in_time, annum): (0, int),
        (population, no_unit): (0, int),
        (number_of_households, no_unit): (0, int),
        (elevation, metre): (0, int),
        }
    return pred_id_and_unit_id2precision_and_type.get(
            (pred_id, unit_id), (None, None))


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
        ('--wikidata-num-attributes-max-year', dict(type=int, default=10000)),
        ('--wikidata-num-attributes-min-year', dict(type=int, default=-10000)),
        ]

    @cached_property
    def attr_labels(self):
        return list(map(
            self.wikidata.property_id2label.get,
            self.pred_enc.labels.tolist()))

    @cached_property
    def attr_id2attr_label(self):
        return {
            k: v for k, v in self.wikidata.property_id2label.items()
            if k in self.allowed_attrs
            }

    def attr_idx2attr_label(self, idx):
        if not hasattr(idx, '__iter__'):
            idx = [idx]
        return list(map(
            self.attr_id2attr_label.__getitem__,
            self.pred_enc.inverse_transform(idx)
            ))

    @cached_property
    def attr_label2attr_id(self):
        d = {v: k for k, v in self.attr_id2attr_label.items()}
        assert len(d) == len(self.attr_id2attr_label)
        return d

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
        f = self.wikidata.data_dir / self.conf.wikidata_unit_labels_fname
        unit_id2label = dict_load(f, splitter='\t')
        unit_id2label['1'] = '1'
        for unit in self.units:
            if unit not in unit_id2label:
                unit_id2label[unit] = unit
        return unit_id2label

    def unit_ids2labels(self, unit_ids):
        return list(map(self.unit_id2label.__getitem__, unit_ids))

    @cached_property
    def units(self):
        return set(self.unit_enc.labels)

    @cached_property
    def n_units(self):
        return len(self.units)

    @staticmethod
    def transform_time(wikidata_time):
        t = wikidata_time
        year = int(t[0] + t[1:].split("-")[0])
        return year

        # a small number of  astronomical and geological events have
        # very large absolute values. Clip these for numerical reasons
        # return min(2100, max(-10000, t))

    def time_is_in_allowed_range(self, wikidata_time):
        year = self.transform_time(wikidata_time)
        min_year = self.conf.wikidata_num_attributes_min_year
        max_year = self.conf.wikidata_num_attributes_max_year
        return min_year <= year <= max_year

    def filter_quantities(self, numvals, units):
        try:
            numvals, units = zip(*[
                (v, u) for v, u in zip(numvals, units) if u in self.units])
        except ValueError:
            return [None], [None]
        most_freq_unit = Counter(units).most_common(1)[0][0]
        numvals, units = zip(*[
            (v, u) for v, u in zip(numvals, units) if u == most_freq_unit])
        return numvals, units

    def aggregate_quantities(self, quantities, aggregate_fn=np.median):
        numvals, units = zip(*quantities)
        unique_units = set(units)
        if len(unique_units) != 1:
            numvals, units = self.filter_quantities(numvals, units)
            unique_units = set(units)
            assert len(set(units)) == 1
        unit = next(iter(unique_units))
        if unit is None:
            numval = None
        else:
            numval = aggregate_fn(list(map(float, numvals)))
        return numval, unit

    def get_numattrs(self, inst):
        # in variable names:
        # p = predicate, q = quantity, v = numeric value, u = unit
        # a quantity is a pair of a numeric value and a unit
        qdict = inst.get('quantity', {})
        if 'time' in inst:
            tdict = inst['time']
            titems = [
                [p, [[self.transform_time(v), self.year_unit]]]
                for p, v in tdict.items()
                if self.time_is_in_allowed_range(v)
                ]
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
        numvals_units = list(map(self.aggregate_quantities, quantitiess))
        attrs, numvals_units = zip(*[
            [p, vu] for p, vu in zip(attrs, numvals_units)
            if vu[1] is not None])
        return attrs, numvals_units

    @cached_property
    def raw(self):
        return [self.get_numattrs(inst) for inst in self.wikidata.raw['train']]

    def of(self, inst, *args, **kwargs):
        props, values_units = self.get_numattrs(inst)
        return dict(
            item
            for p, vu in zip(props, values_units)
            for item in {
                p: True,
                p + '_value': vu[0],
                p + '_unit': vu[1],
                }.items()
            )

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

    @global_cached_property(fname_tpl='unit_enc.{conf_str}.pkl')
    def unit_enc(self):
        unit_enc = LabelEncoder.from_file(
            self.allowed_units_file,
            additional_labels=[self.none_label])
        assert unit_enc.transform(self.none_label) == 0
        return unit_enc

    def sparse_tensors_to_dense(self, sparse_tensors):
        import scipy
        v_raw_csc = sparse_tensors['v_raw_csc']
        v_raw = torch.tensor(v_raw_csc.toarray()).to(dtype=torch.float)
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
        return {'v': v, 'u': u, 'entity_idx': entity_idx, 'v_raw': v_raw}

    @global_cached_property
    def tensor(self):
        return self.sparse_tensors_to_dense(self.sparse_tensors)

    @file_cached_property
    def counts(self):
        return self.counts_with_most_freq_unit

    @file_cached_property
    def _counts_with_any_unit(self):
        return (self.tensor['v'] != self.no_value_sentinel).sum(dim=0)

    @property
    def no_value_sentinel(self):
        return -1

    @property
    def no_unit_sentinel(self):
        return 0

    @file_cached_property
    def counts_with_most_freq_unit(self):
        import scipy.stats
        # scipy.stats.mode can ignore nan
        u = self.tensor['u'].float()
        u[u == self.no_unit_sentinel] = torch.nan
        mode_out = scipy.stats.mode(
            u.numpy(), nan_policy='omit', keepdims=True)
        most_freq_units = mode_out.mode.data.astype(np.int32)
        most_freq_units = torch.tensor(most_freq_units)
        most_freq_unit_mask = self.tensor['u'] == most_freq_units
        assert most_freq_unit_mask.shape == self.tensor['u'].shape
        value_mask = self.tensor['v'] != self.no_value_sentinel
        counts = (value_mask & most_freq_unit_mask).sum(dim=0)
        assert (counts <= self._counts_with_any_unit).all()
        return counts

    def min_count_idx(self, count):
        return (self.counts >= count).int().nonzero()[:, 0]

    def min_count_id(self, count):
        idx = self.min_count_idx(count)
        return self.pred_enc.inverse_transform(idx).tolist()

    def min_count(self, count):
        idx = self.min_count_idx(count)
        return self.attr_idx2attr_label(idx)

    def encode_value(self, value, pred_id):
        pred_idx = self.pred_enc.transform(pred_id)[0]
        value = torch.tensor([value]).reshape(-1, 1)
        value_enc = self.numval_encs[pred_idx].transform(value)[0]
        assert 0 <= value_enc <= 1
        return value_enc

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
                existing_mask = v != self.no_value_sentinel
            if not existing_mask.any():
                return [], existing_mask
            attr_name, numvals, unit_ids = zip(*[
                (attr_name, numval, unit_id)
                for e, attr_name, numval, unit_id in zip(
                    existing_mask, self.attr_names, numvals, unit_ids)
                if e])

        unit_names = list(map(self.unit_id2name.get, unit_ids))
        return list(zip(attr_name, numvals, unit_names)), existing_mask

    def search(self, query, counts=False):
        import re
        pattern = re.compile(query)

        matches = [
            (label, attr_id)
            for label, attr_id in self.attr_label2attr_id.items()
            if re.search(pattern, label)
            ]
        return matches

    def values(self, pred_id, raw=True):
        pred_idx = self.pred_enc(pred_id)[0]
        tensor_key = 'v' + '_raw' if raw else ''
        return self.tensor[tensor_key][:, pred_idx]

    def units_for_pred(self, pred_id):
        pred_idx = self.pred_enc(pred_id)[0]
        return self.tensor['u'][:, pred_idx]

    def value_mask(self, pred_id, most_freq_unit_only=True):
        pred_idx = self.pred_enc(pred_id)[0]
        unit_mask = self.tensor['u'][:, pred_idx] != self.no_unit_sentinel
        value_mask = self.tensor['v'][:, pred_idx] != self.no_value_sentinel
        assert (unit_mask == value_mask).all()
        if most_freq_unit_only:
            unit_idxs = self.tensor['u'][unit_mask, pred_idx].tolist()
            unit_idx = Counter(unit_idxs).most_common(1)[0][0]
            value_mask = self.tensor['u'][:, pred_idx] == unit_idx
        return value_mask

    @property
    def integer_units(self):
        return {
            '1',
            'Q1092296',  # annum
            'Q7727',  # minute
            'Q712226',  # square_km
            }

    @file_cached_property
    def pred_id2std(self):
        return {
            pred_id: df.value_std[0]
            for pred_id, df in self.pred_id2data().items()
            }

    def for_pred(
            self,
            pred_id,
            entity_id=None,
            entity_idx=None,
            entity_label=None,
            raw=True,
            stats=None,
            most_freq_unit_only=True,
            with_entity_popularity=False,
            popularity_quantiles=0,
            precision=None,
            value_type=None,
            lower_value_precision=False,
            **quantile_kwargs
            ):
        if entity_id is None and entity_idx is None:
            mask = self.value_mask(
                pred_id, most_freq_unit_only=most_freq_unit_only)
            entity_id = self.wikidata.entity_id_enc.inverse_transform(entity_idx)
            entity_idx = self.wikidata.entity_idxs[mask]
        elif entity_idx is None:
            entity_idx = self.wikidata.entity_id_enc.transform(entity_id)
        elif entity_id is None:
            entity_id = self.wikidata.entity_id_enc.inverse_transform(entity_idx)
        if entity_label is None:
            entity_label = self.wikidata.entity_ids2labels(entity_id)
        value = self.values(pred_id, raw=raw)[entity_idx]
        unit_idx = self.units_for_pred(pred_id)[entity_idx]
        unit_id = self.unit_enc.inverse_transform(unit_idx)
        unit_label = self.unit_ids2labels(unit_id)
        data = dict(
            entity_idx=entity_idx,
            entity_id=entity_id,
            entity_label=entity_label,
            value=value,
            unit_idx=unit_idx,
            unit_id=unit_id,
            unit_label=unit_label,
            )
        df = pd.DataFrame(data)

        def get_stats_from_df(_df):
            return {
                'value_mean': _df.value.mean(),
                'value_mode': _df.value.mode().mean(),  # mode can return multiple values
                'value_std': _df.value.std(),
                }

        if isinstance(stats, pd.DataFrame):
            stats = get_stats_from_df(stats)
        if stats is None:
            stats = get_stats_from_df(df)
        for k, v in stats.items():
            df[k] = v
        df['value_z_score'] = (df.value - df.value_mean) / df.value_std
        df['value_sort_idx'] = np.argsort(df.value)
        df['value_z_score_sort_idx'] = np.argsort(df.value_z_score)

        if lower_value_precision:
            assert precision is None
            assert value_type is None
            pred_id = str(pred_id)
            unit_id = take_singleton(set(unit_id))
            precision, value_type = precision_and_type_for_pred_and_unit(pred_id, unit_id)
        if precision is not None:
            df['value'] = df['value'].round(precision)
        if value_type is not None:
            df['value'] = df['value'].astype(value_type)

        if with_entity_popularity:
            entity_ids = df.entity_id.tolist()
            popularity_df = self.wikidata.popularity.with_quantiles(
                entity_ids,
                n_quantiles=popularity_quantiles,
                **quantile_kwargs,
                )
            df = pd.concat([df, popularity_df], axis=1)
        return df

    def pred_id2data(
            self,
            n_inst=10000,
            ignored_preds=None,
            with_entity_popularity=True,
            popularity_quantiles=10,
            duplicates='drop',
            lower_value_precision=False,
            ):
        if ignored_preds is None:
            ignored_preds = set()
        freq_pred_mask = self.counts >= n_inst
        freq_pred_idxs = freq_pred_mask.nonzero().view(-1)
        freq_pred_ids = self.pred_enc.inverse_transform(freq_pred_idxs)
        return {
            pred_id: self.for_pred(
                pred_id,
                with_entity_popularity=with_entity_popularity,
                popularity_quantiles=popularity_quantiles,
                duplicates=duplicates,
                lower_value_precision=lower_value_precision,
                )
            for pred_id in freq_pred_ids
            if pred_id not in ignored_preds
            }

    def plot_values(self, pred_id, title=''):
        df = self.for_pred(pred_id)
        from dougu.plot import plot_distribution, simple_plot
        pred_label = self.attr_id2attr_label[pred_id].replace(' ', '_')
        pred_str = pred_id + '_' + pred_label
        for col in ('value', 'value_z_score'):
            _title = title + ('.' if title else '') + f'{pred_str}.{col}'
            print(_title)
            plot_distribution(data=df, x=col, title=_title + '.violin')
            sort_idx = df[f'{col}_sort_idx']
            y = df[col][sort_idx]
            simple_plot(y=y, title=_title + '.line')

    def plot_all_values(self, min_count=1000):
        for pred_id in self.min_count_id(min_count):
            self.plot_values(pred_id)
