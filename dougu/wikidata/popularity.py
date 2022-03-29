import torch

from dougu import (
    cached_property,
    file_cached_property,
    )

from .wikidata_attribute import WikidataAttribute


class WikidataPopularity(WikidataAttribute):
    @cached_property
    def raw(self):
        raise NotImplementedError('todo')

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

        indegree = counts(self.wikidata.relations.tensor[:, 2])
        outdegree = counts(self.wikidata.relations.tensor[:, 0])

        return {
            'indegree': indegree,
            'outdegree': outdegree,
            'sum_in_out': indegree + outdegree,
            }

