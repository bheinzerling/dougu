from pathlib import Path
from dataclasses import dataclass
import random
from functools import (
    cache,
    total_ordering,
    wraps,
    )
from itertools import islice
from collections import deque

import torch

import networkx as nx

from dougu import (
    flatten,
    dict_load,
    groupby,
    cached_property,
    file_cached_property,
    )
from dougu.graph import graph_from_edges

from .wikidata_attribute import WikidataAttribute

max_depth = 999999


@dataclass
@total_ordering
class Node:
    id: str
    label: str = None
    n_descendants: int = 0
    depth: int = max_depth

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        if self.depth == other.depth:
            return self.n_descendants > other.n_descendants
        return self.depth < other.depth


class WikidataTypes(WikidataAttribute):
    key = 'P31'
    args = [
        ('--wikidata-types-fname',
            dict(type=Path, default='P31.obj.mincount_100')),
        ('--wikidata-types-counts-fname',
            dict(type=Path, default='P31.obj.counts')),
        ('--wikidata-types-label-fname',
            dict(type=Path, default='P31.obj.labels_en')),
        ('--wikidata-types-max-graph-nodes', dict(type=int)),
        ('--wikidata-types-min-descendants', dict(type=int, default=0)),
        ('--wikidata-types-random-seed', dict(type=int, default=0)),
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(self.conf.wikidata_types_random_seed)

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'wikidata_types_fname',
            'wikidata_types_max_graph_nodes',
            'wikidata_types_min_descendants',
            ]

    def log_size(self):
        tensor = self.tensor.float()
        self.log(f'{tensor.shape}')
        self.log(f'{tensor.sum(dim=1).mean().item():.2f} types per instance')
        self.log(f'{tensor.sum(dim=1).max().item():.2f} types max')

    @property
    def allowed_types_file(self):
        return self.wikidata.data_dir / self.conf.wikidata_types_fname

    @cached_property
    def allowed_types(self):
        return set(self.type_enc.labels)

    @cached_property
    def n_types(self):
        return len(self.allowed_types)

    @file_cached_property
    def type_enc(self):
        from dougu.codecs import LabelEncoder
        return LabelEncoder.from_file(self.allowed_types_file, to_torch=True)

    def of(self, inst, *args, **kwargs):
        types = set(inst.get(self.key, []))
        return list(filter(self.allowed_types.__contains__, types))

    @cached_property
    def counts(self):
        f = self.wikidata.data_dir / self.conf.wikidata_types_counts_fname
        d = dict_load(f)
        return {k: v for k, v in d.items() if k in self.allowed_types}

    @cached_property
    def raw(self):
        id2types = {
            inst['id']: self.of(inst) for inst in self.wikidata.raw['train']}
        assert set(id2types.keys()) == set(self.entity_ids)
        return [id2types[entity_id] for entity_id in self.entity_ids]

    @file_cached_property
    def tensor(self):
        typess = self.raw
        n_types = torch.tensor(list(map(len, typess)))
        types = list(flatten(typess))
        types_enc = self.type_enc.transform(types)

        entity_idxs = torch.arange(len(typess))
        row_idxs = entity_idxs.repeat_interleave(n_types)
        col_idxs = types_enc
        idxs = torch.stack([row_idxs, col_idxs])
        vals = torch.ones_like(col_idxs, dtype=torch.int8)
        size = torch.Size((len(typess), self.n_types))
        return torch.sparse.ByteTensor(idxs, vals, size).to_dense()

    def tensorize(self):
        return {
            split_name: self.tensorize_split(split)
            for split_name, split in self.raw.items()}

    @property
    def color_tensor_source(self):
        return self.tensor

    @property
    def color_tensor(self):
        from sklearn.decomposition import PCA
        c = PCA(n_components=3).fit_transform(self.color_tensor_source)
        c = (c - c.min()) / c.ptp()
        color = (c * 255).astype(int)
        return color

    @cached_property
    def label2type_ids(self):
        keys, values = zip(*self.type_id2label.items())
        return groupby(values, keys)

    @cached_property
    def type_id2label(self):
        f = self.wikidata.data_dir / self.conf.wikidata_types_label_fname
        d = dict_load(f, splitter='\t')
        return {k: v for k, v in d.items() if k in self.allowed_types}

    def type_idxs2labels(self, type_idxs):
        type_ids = self.type_enc.inverse_transform(type_idxs)
        if isinstance(type_ids, str):
            return self.type_id2label[type_ids]
        return list(map(self.type_id2label.__getitem__, type_ids))

    def search(self, query, counts=False):
        import re
        pattern = re.compile(query)

        def maybe_add_counts(type_ids):
            if counts:
                return [
                    (type_id, self.counts[type_id])
                    for type_id in type_ids
                    ]
            return type_ids

        matches = [
            (label, maybe_add_counts(type_ids))
            for label, type_ids in self.label2type_ids.items()
            if re.search(pattern, label)
            ]
        return matches

    @file_cached_property
    def graph(self):
        def edges(instances):
            preds = {
                'P31': 1000,  # instance of
                'P279': 1,  # subclass of
                }
            k = self.conf.wikidata_types_max_graph_nodes
            for inst in islice(instances, k):
                for pred, weight in preds.items():
                    target = inst['id']
                    rels = inst.get('relation', {})
                    sources = set(rels.get(pred, []))
                    for source in sources:
                        if source == target:
                            continue
                        yield (
                            source, target, {'weight': weight, 'label': pred})

        edge_list = list(edges(self.wikidata.raw_iter))
        g = graph_from_edges(edge_list)
        for node_id, node_data in g.nodes(data=True):
            label = self.wikidata.entity_id2label.get(node_id, node_id)
            node_data['label'] = label
            node_data['title'] = node_id + ' ' + label
        return g

    @property
    def graph_undirected(self):
        return self.graph.to_undirected()

    @property
    def root_type(self):
        return 'Q35120'  # entity

    def prune_graph(self, min_descendants):
        g = self.graph
        if min_descendants > 0:
            print('before pruning', '#nodes', len(g.nodes), '#edges', len(g.edges))
            for node_id, node_data in list(g.nodes(data=True)):
                if node_data['n_descendants'] < min_descendants:
                    g.remove_node(node_id)
            print('after pruning', '#nodes', len(g.nodes), '#edges', len(g.edges))

    @cache
    def depth(self, entity_id):
        root_depth = 1
        try:
            return nx.shortest_path_length(
                self.graph,
                source=self.root_type,
                target=entity_id
                ) + root_depth
        except nx.NetworkXNoPath:
            return max_depth

    @cache
    def path_to_root(self, entity_id):
        return list(reversed(nx.shortest_path(
            self.graph,
            target=entity_id,
            source=self.root_type,
            weight='weight',
            )))

    @cache
    def paths_to_root(self, entity_id, cutoff=5, pretty=True):
        path_iters = nx.all_simple_paths(
            self.graph, self.root_type, entity_id, cutoff=cutoff)
        paths = [list(reversed(p)) for p in path_iters]
        if not pretty:
            return paths
        return list(map(self.to_pretty_path, paths))

    def to_pretty_path(self, path):
        to_labels = self.wikidata.entity_ids2labels
        return ' -> '.join(to_labels(path))

    def shortest_path(self, source_id, target_id, directed=False):
        g = self.graph if directed else self.graph_undirected
        return nx.shortest_path(g, source=source_id, target=target_id)

    def descendants(self, node_id):
        return nx.descendants(self.graph, node_id)

    def n_descendants(self, node_id):
        node = self.graph.nodes[node_id]
        try:
            return node['n_descendants']
        except KeyError:
            node['n_descendants'] = len(self.descendants(node_id))
            return node['n_descendants']

    def sorted_nodes(self, nodes):
        nodes = map(self.to_pretty_node, nodes)
        return sorted(nodes)

    def node_shim(method):
        @wraps(method)
        def wrapper(self, node, sort=True):
            if isinstance(node, Node):
                node = node.id
            nodes = method(self, node)
            if sort:
                nodes = self.sorted_nodes(nodes)
            return nodes
        return wrapper

    @node_shim
    def children(self, node):
        return self.graph.successors(node)

    @node_shim
    def parents(self, node):
        return self.graph.predecessors(node)

    @node_shim
    def ancestors(self, node):
        from collections import deque
        seen = set()
        queue = deque([node])
        while queue:
            node = queue.pop()
            parents = self.graph.predecessors(node)
            for parent in parents:
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return seen

    def to_pretty_node(self, node):
        if isinstance(node, Node):
            return node
        node_id = node
        node = self.graph.nodes[node_id]
        return Node(
            node_id,
            label=node['label'],
            n_descendants=self.n_descendants(node_id),
            depth=self.depth(node_id),
            )

    def sample_descendants_of(
            self,
            node_id,
            sample_size=100,
            exclude_descendants_of=None,
            instances_only=False,
            random_state=None,
            exclude_nodes=None,
            ):
        descendants = self.descendants(node_id)
        if instances_only:
            raise NotImplementedError()
        if exclude_descendants_of:
            descendants -= self.descendants(exclude_descendants_of)
            descendants -= {exclude_descendants_of}
        if exclude_nodes is not None:
            descendants -= exclude_nodes
        if random_state is not None:
            rng = random_state
        else:
            rng = random
        # descendants or unordered, need to sort so that sample is reproducible
        descendants = sorted(descendants)
        return rng.sample(descendants, sample_size)

    def sample_contrastive_groups(
            self,
            group0_ancestor,
            group1_ancestor=None,
            sample_size=100,
            random_state=None,
            exclude_nodes=None,
            ):
        if group1_ancestor is None:
            group1_ancestor = self.root_type
        group0 = self.sample_descendants_of(
            group0_ancestor,
            sample_size=sample_size,
            random_state=random_state,
            exclude_nodes=exclude_nodes,
            )
        group1 = self.sample_descendants_of(
            group1_ancestor,
            sample_size=sample_size,
            exclude_descendants_of=group0_ancestor,
            random_state=random_state,
            exclude_nodes=exclude_nodes,
            )
        # groups should be disjoint
        assert not (set(group0) & set(group1))
        return group0, group1

    def traverse(self, start_node=None, sort=False):
        if start_node is None:
            start_node = self.root_type
        if sort:
            node = self.to_pretty_node(start_node)
        else:
            node = start_node
        queue = deque([node])
        added_nodes = set([node])
        while queue:
            node = queue.popleft()
            yield node
            to_add = [c for c in self.children(node, sort=sort) if c not in added_nodes]
            queue.extend(to_add)
            added_nodes.update(to_add)

    def sample_all_countrastive_groups(
            self, sample_size=100, random_state=None):
        breakpoint()
        contrast_node = self.root_type
        from itertools import islice
        for node in islice(self.traverse(), 20):
            if node.id == contrast_node:
                continue
            groups = self.sample_contrastive_groups(
                group0_ancestor=node.id,
                group1_ancestor=contrast_node,
                sample_size=sample_size,
                random_state=random_state,
                )

    @node_shim
    def cohyponyms(self, node):
        return set(c for p in self.parents(node) for c in self.children(p)) \
            - {self.to_pretty_node(node)}
