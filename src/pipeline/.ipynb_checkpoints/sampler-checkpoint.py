import torch

from src.utils.data_util import tensorize_batch_entities

from ..structure import KnowledgeGraph, NeuralBinaryPredicate
from ..utils import RaggedBatch

class TripleSampler:
    def __init__(self) -> None:
        pass

    def __call__(self, batch_input: torch.Tensor) -> RaggedBatch:
        # assert the batch input is triples
        assert isinstance(batch_input, torch.Tensor)
        assert len(batch_input.shape) == 2
        batch_size, y = batch_input.shape
        assert y == 3

        sizes = torch.ones(batch_size, device=batch_input.device)
        return RaggedBatch(flatten=batch_input, sizes=sizes)

class SubgraphSampler:
    def __init__(self, num_hops, kg: KnowledgeGraph) -> None:
        self.num_hops = num_hops
        self.kg = kg

    def __call__(self, batch_input: torch.Tensor) -> RaggedBatch:
        # assert the batch input is nodes
        assert isinstance(batch_input, torch.Tensor)
        assert len(batch_input.shape) == 1

        subgraph_triples = self.kg.get_subgraph(batch_input,
                                                num_hops=self.num_hops)
        return subgraph_triples


class EFGSampler:
    def __init__(self,
                 num_rounds,
                 beam_size,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate) -> None:
        self.num_rounds = num_rounds
        self.beam_size = beam_size
        self.kg = kg
        self.nbp = nbp

    def _filter_batch_entities_out_of_ragged_triples(
            self,
            ragged_triples: RaggedBatch,
            position: str,
            order: str):
        """
        Given ragged triples, get the head/tail entities with maximum/minimum scores
        Input args:
            - ragged triples
            - position, "head" or "tail"
            - order, "max" or "min"
        output:
            - new_batch_entities
            - new_batch_scores
        """

        assert position in ["head", "tail"]
        assert order in ["max", "min"]

        if position == 'head':
            ragged_entities = ragged_triples.run_ops_on_flatten(
                opfunc=lambda x: x[:, 0])
        else:
            ragged_entities = ragged_triples.run_ops_on_flatten(
                opfunc=lambda x: x[:, -1])

        batch_entities = ragged_entities.to_dense_matrix(
            padding_value=-1)

        ragged_scores = ragged_triples.run_ops_on_flatten(
            opfunc=lambda x: self.nbp.batch_predicate_score(x))

        if order == 'min':
            batch_scores = ragged_scores.to_dense_matrix(
                padding_value=torch.inf)
        else:
            batch_scores = ragged_scores.to_dense_matrix(
                padding_value=-torch.inf)

        batch_size = len(ragged_triples.sizes)
        first_index = torch.arange(batch_size, device=self.device)

        if order == 'min':
            selected_index = torch.argmin(batch_scores, dim=-1)
        else:
            selected_index = torch.argmax(batch_scores, dim=-1)

        selected_entity = batch_entities[first_index, selected_index]
        selected_score  = batch_scores[first_index, selected_index]
        return selected_entity, selected_score

    def _spoiler_act_on_finite_model(self, batch_entities):
        # get triples whose head is new entity
        ragged_head_triples = self.kg.get_neighbor_triples_by_tail(
            batch_entities, filtered=True)
        head_min_entities, head_min_scores = \
            self._filter_batch_entities_out_of_ragged_triples(
                ragged_head_triples, position='head', order='min')

        # get triples whose tail is new entity
        ragged_tail_triples = self.kg.get_neighbor_triples_by_head(
            batch_entities, filtered=True)

        tail_min_entities, tail_min_scores = \
            self._filter_batch_entities_out_of_ragged_triples(
                ragged_tail_triples, position='tail', order='min')

        # aggregate the entities with
        batch_new_entity = torch.where(
            head_min_scores < tail_min_scores,
            head_min_entities, tail_min_entities)
        batch_new_scores = torch.minimum(head_min_scores, tail_min_scores)

        return batch_new_entity, batch_new_scores

    def _spoiler_act_on_neural_model(self, batch_entities, round_mask):
        ragged_head_triples = self.kg.get_non_neightbor_triples_by_tail(
            batch_entities, k=self.beam_size)

        head_max_entities, head_max_scores = \
            self._filter_batch_entities_out_of_ragged_triples(
                ragged_head_triples, position='head', order='max')

        # get triples whose tail is new entity
        ragged_tail_triples = self.kg.get_neighbor_triples_by_head(
            batch_entities, filtered=True)

        tail_max_entities, tail_max_scores = \
            self._filter_batch_entities_out_of_ragged_triples(
                ragged_tail_triples, position='tail', order='max')

        # aggregate the entities
        batch_new_entity = torch.where(
            head_max_scores > tail_max_scores,
            head_max_entities, tail_max_entities)
        batch_new_scores = torch.maximum(head_max_scores, tail_max_scores)

        return batch_new_entity, batch_new_scores

    def _spoiler_step(self, batch_entities):
        fin_ent, fin_score = self._spoiler_act_on_finite_model(batch_entities)
        neu_ent, neu_score = self._spoiler_act_on_neural_model(batch_entities)

        fin_prob = self.nbp.score2prob(fin_score)
        neu_prob = 1 - self.nbp.score2prob(neu_score)

        new_batch_entities = torch.where(
            fin_prob < neu_prob, fin_ent, neu_ent)
        return new_batch_entities

    def __call__(self, batch_input: torch.Tensor) -> RaggedBatch:
        # assert the batch input is nodes
        assert isinstance(batch_input, torch.Tensor)
        assert len(batch_input.shape) == 1
        batch_entities = tensorize_batch_entities(batch_input)

        # play the game
        for round in range(self.num_rounds):
            new_batch_entities = self._spoiler_step(batch_entities)
            batch_entities = torch.cat([batch_entities, new_batch_entities],
                                       dim=-1)

        outputs = self.kg.get_subgraph(batch_entities, num_hops=0)
        return outputs