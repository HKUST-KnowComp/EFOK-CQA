import copy

import torch

from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex


@torch.no_grad()
def solve_EFO1(DNF_formula, relation_matrix, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration):
    torch.cuda.empty_cache()
    with torch.no_grad():
        sub_ans_list = []
        n_entity = relation_matrix[0].shape[0]
        for sub_formula in DNF_formula.formula_list:
            all_candidates = {}
            for term_name in sub_formula.term_dict:
                if sub_formula.has_term_grounded_entity_id_list(term_name):
                    all_candidates[term_name] = torch.zeros(n_entity).to(device)
                    all_candidates[term_name][sub_formula.term_grounded_entity_id_dict[term_name][index]] = 1
                else:
                    all_candidates[term_name] = torch.ones(n_entity).to(device)
            sub_graph_edge, sub_graph_negation_edge = [], []
            for pred in sub_formula.predicate_dict.values():
                pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                                pred.tail.name)
                if pred.negated:
                    sub_graph_negation_edge.append(pred_triples)
                else:
                    sub_graph_edge.append(pred_triples)
            sub_kg_index = KGIndex()
            sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
            sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
            neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
            sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
            sub_ans = solve_conjunctive(sub_kg, neg_kg, relation_matrix,
                                        all_candidates, conjunctive_tnorm, existential_tnorm, 'f', device,
                                        max_enumeration)
            sub_ans_list.append(sub_ans)
        if len(sub_ans_list) == 1:
            return sub_ans_list[0]
        else:
            if conjunctive_tnorm == 'product':
                not_ans = 1 - sub_ans_list[0]
                for i in range(1, len(sub_ans_list)):
                    not_ans = not_ans * (1 - sub_ans_list[i])
                return 1 - not_ans
            elif conjunctive_tnorm == 'Godel':
                final_ans = sub_ans_list[0]
                for i in range(1, len(sub_ans_list)):
                    final_ans = torch.maximum(final_ans, sub_ans_list[i])
                return final_ans
            else:
                raise NotImplementedError


@torch.no_grad()
def construct_matrix_list(head_node, tail_node, sub_graph, neg_sub_graph, relation_matrix_list, conj_tnorm):
    node_pair, reverse_node_pair = (head_node, tail_node), (tail_node, head_node)
    h2t_relation, t2h_relation = sub_graph.ht2r[node_pair], sub_graph.ht2r[reverse_node_pair]
    h2t_negation, t2h_negation = neg_sub_graph.ht2r[node_pair], neg_sub_graph.ht2r[reverse_node_pair]
    transit_matrix_list = []
    for r in h2t_relation:
        transit_matrix_list.append(relation_matrix_list[r])
    for r in t2h_relation:
        transit_matrix_list.append(relation_matrix_list[r].transpose(-2, -1))
    for r in h2t_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].to_dense())
    for r in t2h_negation:
        transit_matrix_list.append(1 - relation_matrix_list[r].transpose(-2, -1).to_dense())
    if conj_tnorm == 'product':
        all_prob_matrix = transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            if all_prob_matrix.is_sparse and not transit_matrix_list[i].is_sparse:
                all_prob_matrix = all_prob_matrix.to_dense().multiply(transit_matrix_list[i])
            else:
                all_prob_matrix = all_prob_matrix.multiply(transit_matrix_list[i])
    elif conj_tnorm == 'Godel':
        all_prob_matrix = transit_matrix_list[0].to_dense() \
            if transit_matrix_list[0].is_sparse else transit_matrix_list[0]
        for i in range(1, len(transit_matrix_list)):
            if transit_matrix_list[i].is_sparse:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i].to_dense())
            else:
                all_prob_matrix = torch.minimum(all_prob_matrix, transit_matrix_list[i])
    else:
        raise NotImplementedError

    if all_prob_matrix.is_sparse:  # n*n sparse matrix or dense matrix (when only one negation edges)
        return all_prob_matrix.to_dense()
    else:
        return all_prob_matrix


@torch.no_grad()
def extend_ans(ans_node, sub_ans_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, relation_matrix,
               leaf_candidate, sub_ans, conj_tnorm, exist_tnorm):
    all_prob_matrix = construct_matrix_list(sub_ans_node, ans_node, sub_graph, neg_sub_graph, relation_matrix,
                                            conj_tnorm)
    if conj_tnorm == 'product':
        all_prob_matrix.mul_(sub_ans.unsqueeze(-1))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, sub_ans.unsqueeze(-1))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = (torch.amax(all_prob_matrix, dim=-2)).squeeze()  # prob*vec is 1*n  matrix
        del all_prob_matrix
    elif exist_tnorm == 'product':
        prob_vec = 1 - torch.prod(1 - all_prob_matrix, dim=-2)
    else:
        raise NotImplementedError
    if conj_tnorm == 'product':
        prob_vec = leaf_candidate * prob_vec
    elif conj_tnorm == 'Godel':
        prob_vec = torch.minimum(leaf_candidate, prob_vec)
    else:
        raise NotImplementedError
    return prob_vec


@torch.no_grad()
def existential_update(leaf_node, adjacency_node, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                       r_matrix_list, leaf_candidates, adj_candidates, conj_tnorm, exist_tnorm) -> dict:
    all_prob_matrix = construct_matrix_list(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                            conj_tnorm)
    if conj_tnorm == 'product':
        all_prob_matrix.mul_(leaf_candidates.unsqueeze(-1))
        all_prob_matrix.mul_(adj_candidates.unsqueeze(-2))
    elif conj_tnorm == 'Godel':
        all_prob_matrix = torch.minimum(all_prob_matrix, leaf_candidates.unsqueeze(-1))
        all_prob_matrix = torch.minimum(all_prob_matrix, adj_candidates.unsqueeze(-2))
    else:
        raise NotImplementedError
    if exist_tnorm == 'Godel':
        prob_vec = torch.amax(all_prob_matrix, dim=-2).squeeze()
    else:
        raise NotImplementedError
    return prob_vec


@torch.no_grad()
def cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                         r_matrix_list, now_candidate_set, conj_tnorm, exist_tnorm, now_variable, device,
                         max_enumeration):
    new_candidate_set = copy.deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        adj_candidate_vec = existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)
        new_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    new_candidate_set.pop(to_cut_node)
    sub_answer = solve_conjunctive(new_sub_graph, new_sub_neg_graph, r_matrix_list, new_candidate_set, conj_tnorm,
                                   exist_tnorm, now_variable, device, max_enumeration)
    return sub_answer


def find_enumerate_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    return_candidate = [None, 100, 100000]
    for node in now_candidate:
        if node == now_variable:
            continue
        adjacency_node_list = list(set.union(*[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
                                               neg_sub_graph.t2h[node]]))
        adjacency_node_num = len(adjacency_node_list)
        candidate_num = torch.count_nonzero(now_candidate[node])
        if not return_candidate[0] or adjacency_node_num < len(return_candidate[1]) or \
                (adjacency_node_num == len(return_candidate[1]) and candidate_num < return_candidate[2]):
            return_candidate = node, adjacency_node_list, candidate_num
    return return_candidate[0], return_candidate[1]


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if node == now_variable:
                return node, list(adjacency_node_set)[0], True
            candidate_num = torch.count_nonzero(now_candidate[node])
            if not return_candidate[0] or candidate_num < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], candidate_num]
    return return_candidate[0], return_candidate[1], False


@torch.no_grad()
def solve_conjunctive(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable, device,
                      max_enumeration):
    n_entity = relation_matrix[0].shape[0]
    if not positive_graph.triples and not negative_graph.triples:
        return now_candidate_set[now_variable]
    if len(now_candidate_set) == 1:
        return now_candidate_set
    now_leaf_node, adjacency_node, being_asked_variable = \
        find_leaf_node(positive_graph, negative_graph, now_candidate_set, now_variable)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_list = [adjacency_node]
        if being_asked_variable:
            next_variable = adjacency_node
            sub_pos_g, sub_neg_g = kg_remove_node(positive_graph, now_leaf_node), \
                                   kg_remove_node(negative_graph, now_leaf_node)
            sub_ans = solve_conjunctive(sub_pos_g, sub_neg_g, relation_matrix, now_candidate_set,
                                        conjunctive_tnorm, existential_tnorm, next_variable, device, max_enumeration)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix,
                                   now_candidate_set[now_leaf_node], sub_ans, conjunctive_tnorm, existential_tnorm)
            return final_ans
        else:
            answer = cut_node_sub_problem(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable, device, max_enumeration)
            return answer
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable)
        if max_enumeration:  # TODO: can not be set to 0, need to update
            easy_candidate = torch.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
            enumeration_num = torch.count_nonzero(now_candidate_set[to_enumerate_node])
            max_enumeration_here = max_enumeration + easy_candidate
            to_enumerate_candidates = torch.argsort(now_candidate_set[to_enumerate_node],
                                                    descending=True)[:min(max_enumeration_here, enumeration_num)]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()
        this_node_candidates = copy.deepcopy(now_candidate_set[to_enumerate_node])
        all_enumerate_ans = torch.zeros((to_enumerate_candidates.shape[0], n_entity)).to(device)
        if to_enumerate_candidates.shape[0] == 0:
            return torch.zeros(n_entity).to(device)
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = torch.zeros_like(now_candidate_set[to_enumerate_node]).to(device)
            candidate_truth_value = this_node_candidates[enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable, device, max_enumeration)
            if conjunctive_tnorm == 'product':
                enumerate_ans = candidate_truth_value * answer
            elif conjunctive_tnorm == 'Godel':
                enumerate_ans = torch.minimum(candidate_truth_value, answer)
            else:
                raise NotImplementedError
            all_enumerate_ans[i] = enumerate_ans
        if existential_tnorm == 'Godel':
            final_ans = torch.amax(all_enumerate_ans, dim=-2)
        else:
            raise NotImplementedError
        return final_ans


@torch.no_grad()
def solve_EFO1_new(DNF_formula, relation_matrix, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration):
    torch.cuda.empty_cache()
    with torch.no_grad():
        sub_ans_list = []
        n_entity = relation_matrix[0].shape[0]
        for sub_formula in DNF_formula.formula_list:
            all_candidates = {}
            for term_name in sub_formula.term_dict:
                if sub_formula.has_term_grounded_entity_id_list(term_name):
                    all_candidates[term_name] = torch.zeros(n_entity).to(device)
                    all_candidates[term_name][sub_formula.term_grounded_entity_id_dict[term_name][index]] = 1
                else:
                    all_candidates[term_name] = torch.ones(n_entity).to(device)
            sub_graph_edge, sub_graph_negation_edge = [], []
            for pred in sub_formula.predicate_dict.values():
                pred_triples = (pred.head.name, sub_formula.pred_grounded_relation_id_dict[pred.name][index],
                                pred.tail.name)
                if pred.negated:
                    sub_graph_negation_edge.append(pred_triples)
                else:
                    sub_graph_edge.append(pred_triples)
            sub_kg_index = KGIndex()
            sub_kg_index.map_entity_name_to_id = {term: 0 for term in sub_formula.term_dict}
            sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
            neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
            sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in sub_formula.predicate_dict}
            sub_ans = solve_conjunctive(sub_kg, neg_kg, relation_matrix,
                                        all_candidates, conjunctive_tnorm, existential_tnorm, 'f1', device,
                                        max_enumeration)
            sub_ans_list.append(sub_ans)
        if len(sub_ans_list) == 1:
            return sub_ans_list[0]
        else:
            if conjunctive_tnorm == 'product':
                not_ans = 1 - sub_ans_list[0]
                for i in range(1, len(sub_ans_list)):
                    not_ans = not_ans * (1 - sub_ans_list[i])
                return 1 - not_ans
            elif conjunctive_tnorm == 'Godel':
                final_ans = sub_ans_list[0]
                for i in range(1, len(sub_ans_list)):
                    final_ans = torch.maximum(final_ans, sub_ans_list[i])
                return final_ans
            else:
                raise NotImplementedError