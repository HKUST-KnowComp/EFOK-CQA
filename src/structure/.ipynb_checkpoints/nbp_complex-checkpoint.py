import torch
from torch import nn

from .neural_binary_predicate import NeuralBinaryPredicate


class ComplEx(NeuralBinaryPredicate, nn.Module):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 scale: float = 1,
                 init_size: float = 1e-3,
                 device = 'cpu', **kwargs):
        super(ComplEx, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.scale = scale

        self._entity_embedding = nn.Embedding(num_entities, 2*embedding_dim)
        self._entity_embedding.weight.data *= init_size

        self._relation_embedding = nn.Embedding(num_relations, 2*embedding_dim)
        self._relation_embedding.weight.data *= init_size

    @property
    def entity_embedding(self):
        return self._entity_embedding.weight

    @property
    def relation_embedding(self):
        return self._relation_embedding.weight


    def embedding_score(self, head_emb, rel_emb, tail_emb):
        # lhs = head_emb[..., :self.rank], head_emb[..., self.rank:]
        # rel = rel_emb[..., :self.rank],  rel_emb[..., self.rank:]
        # rhs = tail_emb[..., :self.rank], tail_emb[..., self.rank:]
        # return torch.sum(
        #     (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
        #     (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
        #     dim=-1)
        est_tail = self.estimate_tail_emb(head_emb, rel_emb)
        return self.entity_pair_scoring(est_tail, tail_emb)

    def score2truth_value(self, score):
        return torch.sigmoid(score / self.scale)

    def estimate_tail_emb(self, head_emb, rel_emb):
        lhs = head_emb[:, :self.embedding_dim], head_emb[:, self.embedding_dim:]
        rel = rel_emb[:, :self.embedding_dim], rel_emb[:, self.embedding_dim:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def estimate_head_emb(self, tail_emb, rel_emb):
        rhs = tail_emb[:, :self.embedding_dim], tail_emb[:, self.embedding_dim:]
        rel = rel_emb[:, :self.embedding_dim], -rel_emb[:, self.embedding_dim:]

        return torch.cat([
            rhs[0] * rel[0] - rhs[1] * rel[1],
            rhs[0] * rel[1] + rhs[1] * rel[0]
        ], 1)

    def estiamte_rel_emb(self, head_emb, tail_emb):
        lhs = head_emb[:, :self.embedding_dim], head_emb[:, self.embedding_dim:]
        rhs = tail_emb[:, :self.embedding_dim], tail_emb[:, self.embedding_dim:]

        return torch.cat([
            lhs[0] * rhs[0] + lhs[1] * rhs[1],
            lhs[0] * rhs[1] - lhs[1] * rhs[0]
        ], 1)

    def get_relation_emb(self, relation_id_or_tensor):
        rel_id = torch.tensor(relation_id_or_tensor, device=self.device)
        return self._relation_embedding(rel_id)

    def get_entity_emb(self, entity_id_or_tensor):
        ent_id = torch.tensor(entity_id_or_tensor, device=self.device)
        return self._entity_embedding(ent_id)

    def entity_pair_scoring(self, emb1, emb2):
        scores = torch.sum(emb1 * emb2, dim=-1)
        return scores

    # def get_all_entity_rankings(self, batch_embedding_input, eval_batch_size=16):
    #     batch_size = batch_embedding_input.size(0)
    #     begin = 0
    #     entity_ranking_list = []
    #     for begin in range(0, batch_size, eval_batch_size):
    #         end = begin + eval_batch_size
    #         eval_batch_embedding_input = batch_embedding_input[begin: end]
    #         eval_batch_embedding_input = eval_batch_embedding_input.unsqueeze(-2)
    #         # batch_size, all_candidates
    #         # ranking score should be the higher the better
    #         # ranking_score[entity_id] = the score of {entity_id}
    #         # ranking_score = - torch.norm(eval_batch_embedding_input - self.entity_embedding, dim=-1)
    #         # ranking_score = self.entity_pair_scoring(eval_batch_embedding_input, self.entity_embedding)
    #         ranking_score = torch.cosine_similarity(eval_batch_embedding_input, self.entity_embedding, dim=-1)
    #         # ranked_entity_ids[ranking] = {entity_id} at the {rankings}-th place
    #         ranked_entity_ids = torch.argsort(ranking_score, dim=-1, descending=True)
    #         # entity_rankings[entity_id] = {rankings} of the entity
    #         entity_rankings = torch.argsort(ranked_entity_ids, dim=-1, descending=False)
    #         entity_ranking_list.append(entity_rankings)

    #     batch_entity_rankings = torch.cat(entity_ranking_list, dim=0)
    #     return batch_entity_rankings

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.embedding_dim], lhs[:, self.embedding_dim:]
        rel = rel[:, :self.embedding_dim], rel[:, self.embedding_dim:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def get_random_entity_embed(self, batch_size):
        return torch.normal(0, 1e-3, (batch_size, self.embedding_dim * 2), device=self.device, requires_grad=True)

    def regularization(self, emb):
        r, i = emb[..., :self.embedding_dim], emb[..., self.embedding_dim:]
        norm_vec =  torch.sqrt(r ** 2 + i ** 2)
        reg = torch.sum(norm_vec ** 3, -1)
        return reg
