import argparse
import os.path as osp
from collections import defaultdict
from copy import deepcopy

import torch
import tqdm
import pandas as pd

from FIT import extend_ans, existential_update, solve_EFO1_new
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer
from src.structure.knowledge_graph import KnowledgeGraph, kg_remove_node
from src.structure.knowledge_graph_index import KGIndex
from src.language.foq import ConjunctiveFormula, DisjunctiveFormula
from QG_EFOX import ranking2metrics, evaluate_batch_joint

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--ckpt", type=str, default='ckpt/FB15k-237/FIT/torch_0.005_0.001.ckpt')
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFOX-final')
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--e_norm", type=str, default='Godel', choices=['Godel', 'product'])
parser.add_argument("--c_norm", type=str, default='product', choices=['Godel', 'product'])
parser.add_argument("--max", type=int, default=0)
parser.add_argument("--max_total", type=int, default=10)
parser.add_argument("--formula", type=str, default=None)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=740)
negation_list = ['(r1(s1,f))&(!(r2(s2,f)))', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))', '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))', '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))']



@torch.no_grad()
def solve_EFOX(conj_formula: ConjunctiveFormula, relation_matrix, conjunctive_tnorm, existential_tnorm, index, device,
               max_enumeration, max_enumeration_total):
    torch.cuda.empty_cache()
    with torch.no_grad():
        n_entity = relation_matrix[0].shape[0]
        all_candidates = {}
        for term_name in conj_formula.term_dict:
            if conj_formula.has_term_grounded_entity_id_list(term_name):
                all_candidates[term_name] = torch.zeros(n_entity).to(device)
                all_candidates[term_name][conj_formula.term_grounded_entity_id_dict[term_name][index]] = 1
            else:
                all_candidates[term_name] = torch.ones(n_entity).to(device)
        sub_graph_edge, sub_graph_negation_edge = [], []
        for pred in conj_formula.predicate_dict.values():
            pred_triples = (pred.head.name, conj_formula.pred_grounded_relation_id_dict[pred.name][index],
                            pred.tail.name)
            if pred.negated:
                sub_graph_negation_edge.append(pred_triples)
            else:
                sub_graph_edge.append(pred_triples)
        sub_kg_index = KGIndex()
        sub_kg_index.map_entity_name_to_id = {term: 0 for term in conj_formula.term_dict}
        sub_kg = KnowledgeGraph(sub_graph_edge, sub_kg_index)
        neg_kg = KnowledgeGraph(sub_graph_negation_edge, sub_kg_index)
        sub_kg_index.map_relation_name_to_id = {predicate: 0 for predicate in conj_formula.predicate_dict}
        free_variable_list = list(conj_formula.free_variable_dict.keys())
        free_variable_list.sort()
        sub_ans_dict = solve_conjunctive_all(sub_kg, neg_kg, relation_matrix,
                                    all_candidates, conjunctive_tnorm, existential_tnorm, free_variable_list, device,
                                    max_enumeration, max_enumeration_total, all_candidates)
        ans_emb_list = [sub_ans_dict[term_name] for term_name in free_variable_list]
        return torch.stack(ans_emb_list, dim=0)


def find_leaf_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable_list):
    """
    Find a leaf node with least possible candidate. The now-asking variable is first.
    """
    return_candidate = [None, None, 0]
    for node in now_candidate:
        adjacency_node_set = set.union(
            *[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
              neg_sub_graph.t2h[node]])
        if len(adjacency_node_set) == 1:
            if node in now_variable_list:
                return node, list(adjacency_node_set)[0], True
            candidate_num = torch.count_nonzero(now_candidate[node])
            if not return_candidate[0] or candidate_num < return_candidate[2]:
                return_candidate = [node, list(adjacency_node_set)[0], candidate_num]
    return return_candidate[0], return_candidate[1], False


def find_enumerate_node(sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph, now_candidate, now_variable_list):
    return_candidate = [None, 100, 100000]
    for node in now_candidate:
        if node in now_variable_list:
            continue
        adjacency_node_list = list(set.union(*[sub_graph.h2t[node], sub_graph.t2h[node], neg_sub_graph.h2t[node],
                                               neg_sub_graph.t2h[node]]))
        adjacency_node_num = len(adjacency_node_list)
        candidate_num = torch.count_nonzero(now_candidate[node])
        if not return_candidate[0] or adjacency_node_num < len(return_candidate[1]) or \
                (adjacency_node_num == len(return_candidate[1]) and candidate_num < return_candidate[2]):
            return_candidate = node, adjacency_node_list, candidate_num
    return return_candidate[0], return_candidate[1]


@torch.no_grad()
def solve_conjunctive_all(positive_graph: KnowledgeGraph, negative_graph: KnowledgeGraph, relation_matrix,
                      now_candidate_set: dict, conjunctive_tnorm, existential_tnorm, now_variable_list, device,
                      max_enumeration, max_enumeration_total, all_candidate_set):
    n_entity = relation_matrix[0].shape[0]
    if not positive_graph.triples and not negative_graph.triples:
        return all_candidate_set
    if len(now_candidate_set) == 1:
        return all_candidate_set
    now_leaf_node, adjacency_node, being_asked_variable = \
        find_leaf_node(positive_graph, negative_graph, now_candidate_set, now_variable_list)
    if now_leaf_node:  # If there exists leaf node in the query graph, always possible to shrink into a sub_problem.
        adjacency_node_list = [adjacency_node]
        if being_asked_variable:
            # next_variable = adjacency_node
            sub_pos_g, sub_neg_g = kg_remove_node(positive_graph, now_leaf_node), \
                                   kg_remove_node(negative_graph, now_leaf_node)
            copy_variable_list = deepcopy(now_variable_list)
            copy_variable_list.remove(now_leaf_node)
            sub_ans_dict = solve_conjunctive_all(sub_pos_g, sub_neg_g, relation_matrix, now_candidate_set,
                                                 conjunctive_tnorm, existential_tnorm, copy_variable_list, device,
                                                 max_enumeration, max_enumeration_total, all_candidate_set)
            final_ans = extend_ans(now_leaf_node, adjacency_node, positive_graph, negative_graph, relation_matrix,
                                   now_candidate_set[now_leaf_node], sub_ans_dict[adjacency_node], conjunctive_tnorm,
                                   existential_tnorm)
            all_candidate_set[now_leaf_node] = final_ans
            return all_candidate_set
        else:
            sub_candidate_set = cut_node_sub_problem(now_leaf_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable_list, device, max_enumeration, max_enumeration_total,
                                                     all_candidate_set)
            return sub_candidate_set
    else:
        to_enumerate_node, adjacency_node_list = find_enumerate_node(positive_graph, negative_graph, now_candidate_set,
                                                                     now_variable_list)
        easy_candidate = torch.count_nonzero(now_candidate_set[to_enumerate_node] == 1)
        enumeration_num = torch.count_nonzero(now_candidate_set[to_enumerate_node])
        max_enumeration_here = min(max_enumeration + easy_candidate, max_enumeration_total)
        if torch.count_nonzero(now_candidate_set[to_enumerate_node]) > 100:
            sub_candidate_set = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                 relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                 now_variable_list, device, max_enumeration, max_enumeration_total, all_candidate_set)
            return sub_candidate_set
        if max_enumeration is not None:
            to_enumerate_candidates = torch.argsort(now_candidate_set[to_enumerate_node],
                                                    descending=True)[:min(max_enumeration_here, enumeration_num)]
        else:
            to_enumerate_candidates = now_candidate_set[to_enumerate_node].nonzero()
        this_node_candidates = deepcopy(now_candidate_set[to_enumerate_node])
        all_enumerate_ans = torch.zeros((to_enumerate_candidates.shape[0], n_entity)).to(device)
        if to_enumerate_candidates.shape[0] == 0:
            return {variable: torch.zeros(n_entity).to(device) for variable in all_candidate_set}
        for i, enumerate_candidate in enumerate(to_enumerate_candidates):
            single_candidate = torch.zeros_like(now_candidate_set[to_enumerate_node]).to(device)
            candidate_truth_value = this_node_candidates[enumerate_candidate]
            single_candidate[enumerate_candidate] = 1
            now_candidate_set[to_enumerate_node] = single_candidate
            answer_dict = cut_node_sub_problem(to_enumerate_node, adjacency_node_list, positive_graph, negative_graph,
                                          relation_matrix, now_candidate_set, conjunctive_tnorm, existential_tnorm,
                                          now_variable_list, device, max_enumeration, max_enumeration_total,
                                               all_candidate_set)
            for free_variable in now_variable_list:
                answer = answer_dict[free_variable]
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
                all_candidate_set[free_variable] = final_ans
        return all_candidate_set


@torch.no_grad()
def cut_node_sub_problem(to_cut_node, adjacency_node_list, sub_graph: KnowledgeGraph, neg_sub_graph: KnowledgeGraph,
                         r_matrix_list, now_candidate_set, conj_tnorm, exist_tnorm, now_variable, device,
                         max_enumeration, max_enumeration_total, all_candidate_set):
    new_candidate_set = deepcopy(now_candidate_set)
    for adjacency_node in adjacency_node_list:
        adj_candidate_vec = existential_update(to_cut_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list,
                                               new_candidate_set[to_cut_node], new_candidate_set[adjacency_node],
                                               conj_tnorm, exist_tnorm)
        new_candidate_set[adjacency_node] = adj_candidate_vec
        all_candidate_set[adjacency_node] = adj_candidate_vec
    new_sub_graph, new_sub_neg_graph = kg_remove_node(sub_graph, to_cut_node), \
                                       kg_remove_node(neg_sub_graph, to_cut_node)
    new_candidate_set.pop(to_cut_node)
    sub_answer = solve_conjunctive_all(new_sub_graph, new_sub_neg_graph, r_matrix_list, new_candidate_set, conj_tnorm,
                                       exist_tnorm, now_variable, device, max_enumeration, max_enumeration_total,
                                       all_candidate_set)
    return sub_answer


def compute_single_evaluation(fof: DisjunctiveFormula, batch_ans_tensor, n_entity, eval_device):
    argsort = torch.argsort(batch_ans_tensor, dim=-1, descending=True)
    ranking = argsort.clone().to(torch.float).to(eval_device)
    ranking = ranking.scatter_(2, argsort, torch.arange(n_entity).to(torch.float).
                               repeat(argsort.shape[0], argsort.shape[1], 1).to(eval_device))
    two_marginal_logs = defaultdict(float)
    one_marginal_logs = defaultdict(float)
    no_marginal_logs = defaultdict(float)
    f_str_list = [f'f{i + 1}' for i in range(len(fof.free_term_dict))]
    f_str = '_'.join(f_str_list)
    if len(fof.free_term_dict) == 1:
        with torch.no_grad():
            ranking.squeeze_()
            for i in range(batch_ans_tensor.shape[0]):
                # ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
                easy_ans = [instance[0] for instance in fof.easy_answer_list[i][f_str]]
                hard_ans = [instance[0] for instance in fof.hard_answer_list[i][f_str]]
                mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, eval_device)
                two_marginal_logs['MRR'] += mrr
                two_marginal_logs['HITS1'] += h1
                two_marginal_logs['HITS3'] += h3
                two_marginal_logs['HITS10'] += h10
            two_marginal_logs['num_queries'] += batch_ans_tensor.shape[0]
            return two_marginal_logs, one_marginal_logs, no_marginal_logs
    else:
        with torch.no_grad():
            two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(
                ranking, fof.easy_answer_list, fof.hard_answer_list, eval_device, f_str)
            return two_marginal_logs, one_marginal_logs, no_marginal_logs


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if 'NELL' in args.data_folder:
        torch.set_default_dtype(torch.float16)
    writer = Writer(case_name=args.ckpt, config=args, log_path='EFO-1_log')
    relation_matrix_list = torch.load(args.ckpt)
    n_relation, n_entity = len(relation_matrix_list), relation_matrix_list[0].shape[0]
    if args.cuda < 0:
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(args.cuda))
    for i in range(len(relation_matrix_list)):
        if 'NELL' in args.data_folder:
            relation_matrix_list[i] = relation_matrix_list[i].to(torch.float16).to(cuda_device)
        else:
            relation_matrix_list[i] = relation_matrix_list[i].to(cuda_device)
    all_metrics = defaultdict(dict)
    all_formula_data = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    for i, row in tqdm.tqdm(all_formula_data.iterrows(), total=len(all_formula_data)):
        formula_id = row['formula_id']
        if i > args.end or i < args.start:
            continue
        formula = row['formula']
        # data_path = osp.join(configure['data']['data_folder'], f'test_type{i:04d}_EFOX_qaa.json')
        formula_path = osp.join(args.data_folder, f'test_{formula_id}_EFOX_qaa.json')
        if not osp.exists(formula_path):
            print(f'Warnings,{formula_path} not exists!')
            continue
        test_dataloader = QueryAnsweringSeqDataLoader_v2(
            formula_path,
            # size_limit=args.batch_size * 1,
            target_lstr=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0)
        fof_list = test_dataloader.get_fof_list_no_shuffle()
        t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
        all_two_log, all_one_log, all_no_log = defaultdict(float), defaultdict(float), defaultdict(float)
        for ifof, fof in t:
            torch.cuda.empty_cache()
            batch_ans_list, metric = [], {}
            for query_index in range(len(fof.easy_answer_list)):
                ans = solve_EFOX(fof.formula_list[0], relation_matrix_list, args.c_norm, args.e_norm, query_index,
                                     cuda_device, args.max, args.max_total)  # ans shape is [X * n_entity], X is number of free variable
                batch_ans_list.append(ans)
            batch_ans_tensor = torch.stack(batch_ans_list, dim=0)  # batch_size * X * n_entity
            log, one_log, nol_log = compute_single_evaluation(fof, batch_ans_tensor, n_entity, cuda_device)
            del batch_ans_tensor
            for metric in log:
                all_two_log[metric] += log[metric]
            for metric in one_log:
                all_one_log[metric] += one_log[metric]
            for metric in nol_log:
                all_no_log[metric] += nol_log[metric]
        '''
        for log_metric in all_two_log.keys():
            if log_metric != 'num_queries':
                all_two_log[log_metric] /= all_two_log['num_queries']
        for log_metric in all_one_log.keys():
            if log_metric != 'num_queries':
                all_one_log[log_metric] /= all_one_log['num_queries']
        for log_metric in all_one_log.keys():
            all_two_log[f'marginal_{log_metric}'] = all_one_log[log_metric]
        '''
        print(all_two_log)
        writer.save_pickle({formula: [all_two_log, all_one_log, all_no_log]}, f"all_logging_test_0_{formula_id}.pickle")
