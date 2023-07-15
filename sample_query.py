import sys
import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph, kg2matrix
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from train_lmpnn import name2lstr, newlstr2name, index2newlstr, index2EFOX_minimal
from src.language.grammar import parse_lstr_to_disjunctive_formula
from src.language.foq import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import (QueryAnsweringMixDataLoader, QueryAnsweringSeqDataLoader,
                            QueryAnsweringSeqDataLoader_v2,
                            TrainRandomSentencePairDataLoader)



query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--double_check", type=float, default=0.01)
parser.add_argument("--output_folder", type=str, default='data/FB15k-EFOX-final')
parser.add_argument("--data_folder", type=str, default='data/FB15k-EFOX-final')
parser.add_argument("--num_positive", type=int, default=800)
parser.add_argument("--num_negative", type=int, default=400)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
parser.add_argument("--meaningful_negation", type=bool, default=True)
parser.add_argument("--negation_tolerance", type=int, default=2)
parser.add_argument("--ncpus", type=int, default=10)
parser.add_argument("--skip_exist", type=bool, default=False)
parser.add_argument("--sample_formula_scope", type=str, default='EFOX', choices=['real_EFO1', 'EFOX_minimal', 'EFOX'])
parser.add_argument("--sample_formula_list", type=list, default=list(range(0, 1)))
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=741)
parser.add_argument("--max_ans", type=int, default=100)
parser.add_argument("--store_each", type=int, default=5)


lstr_3c = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(r5(e1,e2))'
lstr_3pnc = '((((r1(s1,e1))&(r2(e1,f)))&(r3(s2,e2)))&(r4(e2,f)))&(!(r5(e1,e2)))'
lstr_mi = '(((r1(s1,e1))&(r2(e1,f)))&(r3(e1,f)))&(r4(s2,f))'
lstr_2an = '(r1(e1,f))&(!(r2(s1,f)))'
lstr_3pcp = '(((((r1(s1,e1))&(r2(e1,e3)))&(r3(s2,e2)))&(r4(e2,e3)))&(r5(e1,e2)))&(r6(e3,f))'


def double_checking_answer(given_lstr, fof_qa_dict, kg: KnowledgeGraph):
    if kg is None:
        return None
    if given_lstr == lstr_3c:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = kg.hr2t[(e1_c, fof_qa_dict['r5'])].intersection(e2_candidate)
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_3pcp:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        e3_candidate = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = kg.hr2t[(e1_c, fof_qa_dict['r5'])].intersection(e2_candidate)
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            e3_candidate.update(f_final_candidate)
        all_ans = set()
        for e3_c in e3_candidate:
            all_ans.update(kg.hr2t[(e3_c, fof_qa_dict['r6'])])
        return all_ans
    elif given_lstr == lstr_3pnc:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        e2_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r3'])]
        all_ans = set()
        for e1_c in e1_candidate:
            f_candidate_set = kg.hr2t[(e1_c, fof_qa_dict['r2'])]
            e2_c_set = e2_candidate - kg.hr2t[(e1_c, fof_qa_dict['r5'])]
            if e2_c_set:
                f_e2_candidate_set = set.union(*[kg.hr2t[(e2_c, fof_qa_dict['r4'])] for e2_c in e2_c_set])
            else:
                f_e2_candidate_set = {}
            f_final_candidate = f_candidate_set.intersection(f_e2_candidate_set)
            all_ans.update(f_final_candidate)
        return all_ans
    elif given_lstr == lstr_mi:
        e1_candidate = kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r1'])]
        f_candidate = kg.hr2t[(fof_qa_dict['s2'], fof_qa_dict['r4'])]
        if e1_candidate:
            f_candidate2 = set.union(
                *[kg.hr2t[(e1_c, fof_qa_dict['r2'])].intersection(kg.hr2t[(e1_c, fof_qa_dict['r3'])])
                  for e1_c in e1_candidate])
        else:
            f_candidate2 = {}
        return f_candidate.intersection(f_candidate2)
    elif given_lstr == lstr_2an:
        f_candidate = kg.r2t[fof_qa_dict['r1']]
        f_candidate = f_candidate - kg.hr2t[(fof_qa_dict['s1'], fof_qa_dict['r2'])]
        return f_candidate
    else:
        return None


def sample_one_formula_query(given_lstr, part_kg: KnowledgeGraph, full_kg: KnowledgeGraph, num_samples, sample_mode,
                             meaningful_negation, double_checking, negation_tolerance, full_matrix=None, n_cpus: int = 1, max_ans=None,
                             existing_all_qa_dict=None):
    """
    The double-checking have two probabilities: 1. Use Manually write code, 2. use the solver to check.
    Negation tolerance helps to mitigate the requirement of meaningful negation.
    We note this is only for sample queries that are of conjunctive query.
    """
    if num_samples == 0:
        return []

    fof = parse_lstr_to_disjunctive_formula(given_lstr)
    free_variable_list = list(fof.free_term_dict.keys())
    free_variable_list.sort()
    f_str = '_'.join(free_variable_list)
    stored_qa_dict = existing_all_qa_dict if existing_all_qa_dict else set()
    all_query_list = []
    now_index = -1
    full_matrix = full_matrix if full_matrix is not None else kg2matrix(full_kg)
    use_max_ans = len(free_variable_list) * max_ans if max_ans else None
    sample_max_ans = use_max_ans if sample_mode == 'train' else 3 * max_ans
    with tqdm.tqdm(total=num_samples) as pbar:
        while pbar.n < num_samples:
            qa_dict = None
            now_negation_tolerance = 0
            while qa_dict is None:
                if meaningful_negation and negation_tolerance:
                    if now_negation_tolerance >= negation_tolerance:
                        qa_dict, full_answer, epfo_constraint = fof.sample_query(full_kg, False, full_matrix, sample_max_ans)
                        now_negation_tolerance = 0
                    else:
                        qa_dict, full_answer, epfo_constraint = fof.sample_query(full_kg, True, full_matrix, sample_max_ans)
                        now_negation_tolerance += 1
                else:  # Not meaningful negation or not negation tolerance.
                    qa_dict, full_answer, epfo_constraint = fof.sample_query(full_kg, meaningful_negation, full_matrix, sample_max_ans)
            if qa_dict and str(qa_dict) not in stored_qa_dict:  # We notice sampling may fail and return None
                stored_qa_dict.add(str(qa_dict))  # remember it to avoid repeat
                fof.append_qa_instances(qa_dict)
                now_index += 1
                if sample_mode == 'train':
                    if full_answer is None:
                        full_answer = fof.formula_list[0].deterministic_query_set_with_initialization(
                            now_index, full_kg, None, False, epfo_constraint)
                    easy_answer = set()
                else:
                    if full_answer is None:
                        full_answer = fof.formula_list[0].deterministic_query_set_with_initialization(
                            now_index, full_kg, None, False, epfo_constraint)
                    easy_answer = fof.formula_list[0].deterministic_query_set_with_initialization(
                        now_index, part_kg, None, False, epfo_constraint)
                if full_answer - easy_answer and (max_ans is None or len(full_answer - easy_answer) <= use_max_ans):
                    if random.random() < double_checking:
                        check_easy_ans = fof.deterministic_query(now_index, part_kg, 'solver')
                        check_full_ans = fof.deterministic_query(now_index, full_kg, 'solver')
                    else:
                        check_easy_ans, check_full_ans = None, None
                    if check_full_ans is not None:
                        assert full_answer == check_full_ans, \
                            f"In {sample_mode}, the full {fof.lstr, qa_dict, full_answer}, solver ans {check_full_ans}"
                    if check_easy_ans is not None:
                        assert easy_answer == check_easy_ans, \
                            f"In {sample_mode}, the easy {fof.lstr, qa_dict, full_answer}, solver ans {check_full_ans}"
                    if sample_mode == 'train':
                        new_query = [qa_dict, {f_str: list(full_answer)}, []]
                    else:
                        new_query = [qa_dict, {f_str: list(easy_answer)}, {f_str: list(full_answer - easy_answer)}]
                    all_query_list.append(new_query)
                    pbar.update(1)
    return all_query_list, stored_qa_dict


def check_sampled(lstr, qa_dict, part_ans_dict, hard_ans_dict, part_kg: KnowledgeGraph, full_kg: KnowledgeGraph):
    fof = parse_lstr_to_disjunctive_formula(lstr)
    negation_edge = []
    for pred in fof.predicate_dict.values():
        if pred.negated:
            negation_edge.append(pred.name)
    free_variable_list = list(fof.free_term_dict.keys())
    f_str = '_'.join(free_variable_list)
    fof.append_qa_instances(qa_dict)
    part_ans_list, hard_ans_list = part_ans_dict[f_str], hard_ans_dict[f_str]
    part_ans, hard_ans = set([tuple(ans) for ans in part_ans_list]), set([tuple(ans) for ans in hard_ans_list])
    check_part_ans, check_full_ans = fof.deterministic_query(0, part_kg), fof.deterministic_query(0, full_kg)
    assert check_part_ans == part_ans, f"part_ans {part_ans} != {check_part_ans}"
    hard_ans.update(part_ans)
    assert check_full_ans == hard_ans, f"full_ans {hard_ans} != {check_full_ans}"
    if negation_edge:
        not_negation_ans = fof.formula_list[0].deterministic_query_set(0, full_kg, negation_edge)
        assert not_negation_ans != check_full_ans
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    valid_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'valid_kg.tsv'),
        kgindex=kgidx)
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    """
    for lstr in DNF_lstr2name:
        test_sample_query(lstr, train_kg)

    """
    if args.sample_formula_scope == 'EFOX_minimal':
        formula_scope = index2EFOX_minimal
    elif args.sample_formula_scope == 'EFOX':
        formula_scope = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    elif args.sample_formula_scope == 'real_EFO1':
        formula_scope = index2newlstr
    else:
        raise NotImplementedError
    '''
    for index, lstr_index in enumerate(args.sample_formula_list):
        lstr = formula_scope[lstr_index]
        now_data = {lstr: []}
        output_file_name = osp.join(args.output_folder,
                                    f'{args.mode}_{lstr_index}_{args.sample_formula_scope}_qaa.json')
        if os.path.exists(output_file_name):
            with open(output_file_name, 'rt') as f:
                old_data = json.load(f)
        else:
            old_data = {}
        useful_num = 0
        all_qa_dict = set()
        if lstr in old_data:
            for i in range(len(old_data[lstr])):
                if old_data[lstr][i][2]['f'] and str(old_data[lstr][i][0]) not in all_qa_dict:
                    now_data[lstr].append(old_data[lstr][i])
                    useful_num += 1
                all_qa_dict.add(str(old_data[lstr][i][0]))
        if useful_num == args.num_samples:
            continue
        else:
            if args.mode == 'train':
                all_query = sample_one_formula_query(lstr, None, train_kg, args.num_samples - useful_num, args.mode,
                                                     args.meaningful_negation, args.double_check, all_qa_dict)
            elif args.mode == 'valid':
                all_query = sample_one_formula_query(lstr, train_kg, valid_kg, args.num_samples - useful_num, args.mode,
                                                     args.meaningful_negation, args.double_check, all_qa_dict)
            elif args.mode == 'test':
                all_query = sample_one_formula_query(lstr, valid_kg, test_kg, args.num_samples - useful_num, args.mode,
                                                     args.meaningful_negation, args.double_check, all_qa_dict)
            else:
                raise NotImplementedError
            now_data[lstr].extend(all_query)
        with open(output_file_name, 'wt') as f:
            json.dump(now_data, f)
    '''
    all_data = {}
    for i, row in tqdm.tqdm(formula_scope.iterrows(), total=len(formula_scope)):
        if i > args.end_index or i < args.start_index:
            continue
        lstr = row.formula
        fid = row.formula_id
        output_file_name = osp.join(args.output_folder,
                                    f'{args.mode}_{fid}_{args.sample_formula_scope}_qaa.json')
        useful_num = 0
        all_qa_dict = set()
        now_data = {lstr: []}
        if os.path.exists(output_file_name):
            if args.skip_exist:
                continue
            with open(output_file_name, 'rt') as f:
                old_data = json.load(f)
        else:
            old_data = {lstr: []}
        if lstr in old_data:
            for i in range(len(old_data[lstr])):
                if str(old_data[lstr][i][0]) not in all_qa_dict:
                    now_data[lstr].append(old_data[lstr][i])
                    useful_num += 1
                all_qa_dict.add(str(old_data[lstr][i][0]))
        '''
        exist_lstr = list(old_data.keys())[0]
        for i in range(len(old_data[exist_lstr])):
            if str(old_data[exist_lstr][i][0]) not in all_qa_dict:
                qa_dict, easy_answer, hard_answer = old_data[exist_lstr][i]
                check_sampled(lstr, qa_dict, easy_answer, hard_answer, valid_kg, test_kg)
                now_data[lstr].append(old_data[exist_lstr][i])
                useful_num += 1
            all_qa_dict.add(str(old_data[exist_lstr][i][0]))
        '''
        # now_data[lstr] = old_data[lstr]
        print(f'sampling query of {lstr}')
        if args.mode == 'train':
            use_full_matrix = kg2matrix(train_kg)
            all_query = sample_one_formula_query(lstr, None, train_kg, args.num_positive - useful_num, args.mode,
                                                 args.meaningful_negation, args.double_check, args.negation_tolerance,
                                                 use_full_matrix, args.ncpus, args.max_ans, all_qa_dict)

        elif args.mode == 'valid':
            use_full_matrix = kg2matrix(valid_kg)
            all_query = sample_one_formula_query(lstr, train_kg, valid_kg, args.num_positive - useful_num, args.mode,
                                                 args.meaningful_negation, args.double_check, args.negation_tolerance,
                                                 use_full_matrix, args.ncpus, args.max_ans, all_qa_dict)

        elif args.mode == 'test':
            use_full_matrix = kg2matrix(test_kg)
            if '!' in lstr:
                for j in range(0, args.num_negative - useful_num, args.store_each):
                    all_query, all_qa_dict = sample_one_formula_query(lstr, valid_kg, test_kg, args.store_each,
                                                         args.mode,
                                                         args.meaningful_negation, args.double_check,
                                                         args.negation_tolerance,
                                                         use_full_matrix, args.ncpus, args.max_ans, all_qa_dict)
                    now_data[lstr].extend(all_query)
                    with open(output_file_name, 'wt') as f:
                        json.dump(now_data, f)
                    print("now data length: ", len(now_data[lstr]))

            else:
                for j in range(0, args.num_positive - useful_num, args.store_each):
                    all_query, all_qa_dict = sample_one_formula_query(
                        lstr, valid_kg, test_kg, args.store_each, args.mode, args.meaningful_negation,
                        args.double_check, args.negation_tolerance, use_full_matrix, args.ncpus, args.max_ans,
                        all_qa_dict)
                    now_data[lstr].extend(all_query)
                    with open(output_file_name, 'wt') as f:
                        json.dump(now_data, f)
                    print("now data length: ", len(now_data[lstr]))
        else:
            raise NotImplementedError
        all_data[lstr] = now_data[lstr]
        with open(output_file_name, 'wt') as f:
            json.dump(now_data, f)
    # with open(osp.join(args.output_folder, f'{args.mode}_{args.sample_formula_scope}_qaa.json'), 'wt') as f:
    #    json.dump(all_data, f)



