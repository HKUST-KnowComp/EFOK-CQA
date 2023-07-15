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


from src.structure.knowledge_graph import KnowledgeGraph, kg2matrix
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from train_lmpnn import name2lstr, newlstr2name, index2newlstr, index2EFOX_minimal
from src.language.grammar import parse_lstr_to_disjunctive_formula
from src.language.foq import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import (QueryAnsweringMixDataLoader, QueryAnsweringSeqDataLoader,
                            QueryAnsweringSeqDataLoader_v2,
                            TrainRandomSentencePairDataLoader)



train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--output_folder", type=str, default='data/NELL-EFOX-filtered')
parser.add_argument("--data_folder", type=str, default='data/NELL-EFOX-filtered')
parser.add_argument("--num_positive", type=int, default=1000)
parser.add_argument("--num_negative", type=int, default=500)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
parser.add_argument("--skip_exist", type=bool, default=True)
parser.add_argument("--sample_formula_scope", type=str, default='EFOX', choices=['real_EFO1', 'EFOX_minimal', 'EFOX'])
parser.add_argument("--start_index", type=int, default=250)
parser.add_argument("--end_index", type=int, default=740)
parser.add_argument("--double_check_num", type=int, default=100)
parser.add_argument("--use_constraint", type=bool, default=True)


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

    formula_scope = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    problem_list = []
    for i, row in tqdm.tqdm(formula_scope.iterrows(), total=len(formula_scope)):
        if i > args.end_index or i < args.start_index:
            continue
        lstr = row.formula
        fid = row.formula_id
        output_file_name = osp.join(args.output_folder,
                                    f'{args.mode}_{fid}_{args.sample_formula_scope}_qaa.json')
        useful_num = 0
        all_qa_dict = set()
        if os.path.exists(output_file_name):
            with open(output_file_name, 'rt') as f:
                old_data = json.load(f)
            sample_list = random.sample(old_data[lstr], args.double_check_num)
            check_problem = False
            for sample_instance in sample_list:
                qa_dict, easy_ans_dict, hard_ans_dict = sample_instance
                fof = parse_lstr_to_disjunctive_formula(lstr)
                free_variable_list = list(fof.free_term_dict.keys())
                free_variable_list.sort()
                f_str = '_'.join(free_variable_list)
                easy_ans_set = set(tuple(ans) for ans in easy_ans_dict[f_str])
                hard_ans_set = set(tuple(ans) for ans in hard_ans_dict[f_str])
                fof.append_qa_instances(qa_dict)
                if 'f2' in lstr or args.use_constraint:
                    neglect_negation_list = []
                    for pred in fof.predicate_dict.values():
                        if pred.negated:
                            neglect_negation_list.append(pred.name)
                    assert len(neglect_negation_list) <= 1
                    if len(neglect_negation_list) != 1:
                        break
                    neg_pred = fof.predicate_dict[neglect_negation_list[0]]
                    nh, nt = neg_pred.head.name, neg_pred.tail.name
                    if ('s' in nh and 'e' in nt) or ('e' in nh and 's' in nt):
                        print(fid, 'need to check')
                        efpo_constraint = fof.formula_list[0].deterministic_query_set(0, test_kg, neglect_negation_list,
                                                                                      True)
                        skip_ans = fof.formula_list[0].deterministic_query_set_with_initialization(0, test_kg, neglect_negation_list,
                                                                                                   False,
                                                                                                   efpo_constraint)
                        full_ans = fof.formula_list[0].deterministic_query_set_with_initialization(0, test_kg, None,
                                                                                                   False,
                                                                                                   efpo_constraint)
                        part_ans = fof.formula_list[0].deterministic_query_set_with_initialization(0, valid_kg, None,
                                                                                                   False,
                                                                                                   efpo_constraint)
                    else:
                        continue
                else:
                    continue
                    full_ans = fof.formula_list[0].deterministic_query(0, test_kg)
                    part_ans = fof.formula_list[0].deterministic_query(0, valid_kg)

                hard_ans = full_ans - part_ans
                if part_ans == easy_ans_set and hard_ans == hard_ans_set:
                    pass
                else:
                    problem_list.append((lstr, fid, qa_dict))
                    print('problem', lstr, fid, qa_dict)
                    break
                if skip_ans == full_ans:
                    pass
                else:  #  We have make sure the answer is correct by skip_ans is not the same as full_ans.
                    break
        else:
            print('warning, no file', output_file_name)
    if len(problem_list) > 0:
        print(problem_list)
