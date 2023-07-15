from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)


class FITEstimator(AppFOQEstimator):
    def __init__(self, n_entity, n_relation, matrix_ckpt_path, conj_tnorm, exist_tnorm, negative_sample_size, device):
        super().__init__()
        self.name = 'FIT'
        self.device = device
        self.n_entity = n_entity
        self.n_relation = n_relation
        relation_matrix_list = torch.load(matrix_ckpt_path)
        for i in range(len(relation_matrix_list)):
            relation_matrix_list[i] = relation_matrix_list[i].to(device)
        self.matrix_list = relation_matrix_list
        self.conj_tnorm, self.exist_tnorm = conj_tnorm, exist_tnorm

    def get_entity_embedding(self, entity_ids: torch.Tensor):
        emb = torch.zeros(self.n_entity, device=self.device)
        emb[entity_ids] = 1
        return emb

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        rel_matrix = self.matrix_list[proj_ids].to_dense()
        if self.conj_tnorm == 'product':
            rel_matrix.mul_(emb.unsqueeze(-1))
        elif self.conj_tnorm == 'Godel':
            rel_matrix = torch.minimum(rel_matrix, emb.unsqueeze(-1))
        else:
            raise NotImplementedError
        if self.exist_tnorm == 'Godel':
            prob_vec = torch.amax(rel_matrix, dim=-2).squeeze()
        else:
            raise NotImplementedError
        return prob_vec

    def get_negation_embedding(self, embedding: torch.Tensor):
        embedding = 1 - embedding
        return embedding

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        all_emb = torch.stack(conj_emb)
        if self.conj_tnorm == 'product':
            emb = torch.prod(all_emb, dim=0)
        elif self.conj_tnorm == 'Godel':
            emb = torch.min(all_emb, dim=0)
        else:
            raise NotImplementedError
        return emb

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor]):
        return torch.stack(disj_emb, dim=0)

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor,
                                 **kwargs):
        assert False, 'Do not use d in FIT'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        assert False, 'Do not use D in FIT'

    def criterion(self, pred_emb: torch.Tensor, answer_set: List[IntList], union: bool = False):
        pass

    def compute_logit(self, entity_embedding, query_embedding):
        pass

    def compute_all_entity_logit(self, pred_emb: torch.Tensor, union: bool = False) -> torch.Tensor:
        if union:
            if self.conj_tnorm == 'product':
                no_ans_emb = 1 - pred_emb
                ans_emb = 1 - torch.prod(no_ans_emb, dim=0)
            elif self.conj_tnorm == 'Godel':
                ans_emb = torch.max(pred_emb, dim=0)
            else:
                raise NotImplementedError
        else:
            ans_emb = pred_emb
        return ans_emb.unsqueeze(0)  # returns [1, n_entity] tensor, like other estimators.
