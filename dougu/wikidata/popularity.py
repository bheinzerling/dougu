import torch

from dougu import (
    cached_property,
    file_cached_property,
    json_load,
    )

from .wikidata_attribute import WikidataAttribute


class WikidataPopularity(WikidataAttribute):
    @file_cached_property
    def raw(self):
        return [self.of(inst) for inst in self.wikidata.raw['train']]

    def of(self, inst, *args, **kwargs):
        idx = self.wikidata.entity_id_enc.transform(inst['id']).item()
        return {key: tensor[idx].item() for key, tensor in self.tensor.items()}

    @cached_property
    def article_lengths(self):
        qid2article_len = json_load(self.conf.wikidata_article_length_file)
        return [
            qid2article_len.get(qid, 0)
            for qid in self.wikidata.entity_id_enc.labels
            ]

    @file_cached_property
    def tensor(self):

        def counts(node_ids):
            node_ids = torch.sort(node_ids).values
            node_ids, counts = torch.unique_consecutive(
                node_ids, return_counts=True)
            all_counts = torch.zeros(
                self.wikidata.n_entities, dtype=counts.dtype)
            all_counts[node_ids] = counts
            return all_counts

        article_length = torch.tensor(self.article_lengths)
        indegree = counts(self.wikidata.relations.tensor[:, 2])
        outdegree = counts(self.wikidata.relations.tensor[:, 0])

        return {
            'indegree': indegree,
            'outdegree': outdegree,
            'degree': indegree + outdegree,
            'article_length': article_length,
            }

    def with_quantiles(self, entity_ids, n_quantiles=10, **quantile_kwargs):
        import pandas as pd
        popularities = self[entity_ids]
        popularity_df = pd.DataFrame(popularities).add_prefix('popularity_')
        del popularity_df['popularity_outdegree']
        del popularity_df['popularity_indegree']
        zero_start_offset = 1
        if n_quantiles > 0:
            sort_idx_colnames = []
            quantile_colnames = []
            for colname in popularity_df.columns:
                idx_quantile_colname = colname + '_idx_quantile'

                # Use sort indices (obtained via argsort()) instead of
                # raw values in order to break ties. This ensures that
                # we get the desired number of quantiles
                sort_idx_col = popularity_df[colname].argsort()

                idx_quantile_col = pd.qcut(
                    sort_idx_col,
                    n_quantiles,
                    labels=False,
                    **quantile_kwargs,
                    )
                popularity_df[idx_quantile_colname] = idx_quantile_col

                quantile_colname = colname + '_quantile'
                quantile_colnames.append(quantile_colname)
                quantile_col = pd.qcut(
                    popularity_df[colname],
                    n_quantiles,
                    labels=False,
                    **quantile_kwargs,
                    )
                n_missing_quantiles = n_quantiles - quantile_col.max() - zero_start_offset
                quantile_col[quantile_col != 0] += n_missing_quantiles
                assert quantile_col.max() == n_quantiles - zero_start_offset
                popularity_df[quantile_colname] = quantile_col

                sort_idx_colname = colname + '_sort_idx'
                sort_idx_colnames.append(sort_idx_colname)
                popularity_df[sort_idx_colname] = sort_idx_col

            sort_idx_sum = popularity_df[sort_idx_colnames].sum(axis=1)
            idx_sum_quantile = pd.qcut(
                sort_idx_sum,
                n_quantiles,
                labels=False,
                **quantile_kwargs,
                )
            popularity_df['popularity_mean_idx_quantile'] = idx_sum_quantile

            quantile_sum = popularity_df[quantile_colnames].sum(axis=1)
            sum_quantile = pd.qcut(
                quantile_sum,
                n_quantiles,
                labels=False,
                **quantile_kwargs,
                )

            n_missing_quantiles = n_quantiles - sum_quantile.max() - zero_start_offset
            sum_quantile[sum_quantile != 0] += n_missing_quantiles
            popularity_df['popularity_mean_quantile'] = sum_quantile

        return popularity_df
