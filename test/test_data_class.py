import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.structure import get_nbp_class
from src.structure.geometric_graph import QueryGraph
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.utils.data_util import RaggedBatch
from train_lmpnn import name2lstr, newlstr2name, lstr2name, DNF_lstr2name, EFOXlstr
from src.language.grammar import parse_lstr_to_lformula, parse_lstr_to_lformula_v2, DNF_Transformation, concate_iu_chains, parse_lstr_to_disjunctive_formula
from src.language.foq import Disjunction, ConjunctiveFormula, DisjunctiveFormula
from src.utils.data import QueryAnsweringSeqDataLoader, QueryAnsweringSeqDataLoader_v2


data_folder = 'data/FB15k-237-betae'
train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
batch_size = 20

"""
for name, lstr in name2lstr.items():
    formula = parse_lstr_to_lformula(lstr)
    formula_v2 = parse_lstr_to_lformula_v2(lstr)
    DNF_formula = DNF_Transformation(formula_v2)
    concate_formula = concate_iu_chains(formula_v2)
    print(formula.lstr(), formula_v2.lstr(), DNF_formula.lstr(), concate_formula.lstr())
    formula_v2_check = parse_lstr_to_lformula_v2(formula_v2.lstr())
    assert formula_v2_check.lstr() == formula_v2.lstr()


lstr2name = {}
for name, lstr in name2lstr.items():
    formula = parse_lstr_to_lformula_v2(lstr)
    formula = concate_iu_chains(formula)
    if isinstance(formula, Disjunction):
        formula_list = formula.formulas
    else:
        formula_list = [formula]
    conjunctive_formulas_list = [ConjunctiveFormula(formula) for formula in formula_list]
    fof = DisjunctiveFormula(conjunctive_formulas_list)
    print(fof.lstr)
    lstr2name[fof.lstr] = name
print(lstr2name)
"""


def test_parse_formula(given_lstr):
    lformula = parse_lstr_to_lformula_v2(given_lstr)
    disjunctive_formula = parse_lstr_to_disjunctive_formula(given_lstr)
    return lformula.lstr(), disjunctive_formula.lstr


def test_deterministic_query(data_loader: QueryAnsweringSeqDataLoader_v2, kg: KnowledgeGraph, method, device='cpu', pass_lstr=None):
    fofs = data_loader.get_fof_list_no_shuffle()
    with tqdm.tqdm(fofs) as t:
        for fof in t:
            if pass_lstr and fof.lstr in pass_lstr:
                continue
            for i, pos_answer_dict in enumerate(fof.easy_answer_list):
                easy_answer = pos_answer_dict['f']
                tuple_ans_set = set()
                for ans in easy_answer:
                    tuple_ans_set.add((ans,))
                search_answer = fof.deterministic_query(i, kg, method, device)
                assert tuple_ans_set == search_answer, f"We show the fof is {i, fof.lstr, fof.easy_answer_list[i], fof.pred_grounded_relation_id_dict, fof.term_grounded_entity_id_dict}, while the search ans is {search_answer}"
            print(f'batch formula of {fof.lstr} verified')


def test_deterministic_query_instance(lstr, qa_dict, kg: KnowledgeGraph):
    """
    Test the deterministic query by given an instance
    """
    fof_instance = parse_lstr_to_disjunctive_formula(lstr)
    fof_instance.append_qa_instances(qa_dict)
    set_ans = fof_instance.deterministic_query(0, kg, 'set')
    brutal_set_ans = fof_instance.deterministic_query(0, kg, 'brutal_set')
    solver_ans = fof_instance.deterministic_query(0, kg, 'solver')
    assert set_ans == brutal_set_ans == solver_ans


def test_sample_query(given_lstr, kg: KnowledgeGraph, meaningful_negation):
    lformula = parse_lstr_to_lformula_v2(given_lstr)
    lformula = concate_iu_chains(lformula)
    if isinstance(lformula, Disjunction):
        formula_list = lformula.formulas
    else:
        formula_list = [lformula]
    conjunctive_formulas_list = [ConjunctiveFormula(formula) for formula in formula_list]
    fof = DisjunctiveFormula(conjunctive_formulas_list)
    qa_dict = fof.sample_query(kg, meaningful_negation)
    print(qa_dict)
    fof.append_qa_instances(qa_dict)
    print(fof.deterministic_query(0, kg))


def test_query_graph(conj_formula):
    query_graph = QueryGraph(conj_formula)


if __name__ == "__main__":

    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    """
    for lstr in DNF_lstr2name:
        test_sample_query(lstr, train_kg)
    
    for lstr in newlstr2name:
        test_sample_query(lstr, train_kg)
    
    for lstr in DNF_lstr2name:
        formula_lstr, disjunctive_lstr = test_parse_formula(lstr)
        print(lstr, formula_lstr, disjunctive_lstr, lstr == formula_lstr, formula_lstr == disjunctive_lstr)
    for lstr in newlstr2name:
        formula_lstr, disjunctive_lstr = test_parse_formula(lstr)
        print(lstr, formula_lstr, disjunctive_lstr, lstr == formula_lstr, formula_lstr == disjunctive_lstr)
    for lstr in EFOXlstr:
        formula_lstr, disjunctive_lstr = test_parse_formula(lstr)
        print(lstr, formula_lstr, disjunctive_lstr, lstr == formula_lstr, formula_lstr == disjunctive_lstr)
        test_sample_query(lstr, train_kg, True)
    
    """
    qa_dict = {'r1': 132, 'r2': 30, 's1': 8254, 'r3': 389}
    lstr = '(r1(s1,f1))&((r2(f1,f2))&(!(r3(f1,f2))))'
    test_deterministic_query_instance(lstr, qa_dict, test_kg)



"""
train_dataloader = QueryAnsweringSeqDataLoader_v2(
        osp.join(data_folder, 'train-qaa.json'),
        size_limit=20,
        target_lstr=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    train_ckpt = torch.load('sparse/237/torch_train_perfect.ckpt')
    cuda_device = torch.device('cuda:{}'.format(0))
    for i in range(len(train_ckpt)):
        train_ckpt[i] = train_ckpt[i].to(cuda_device)
    time1 = time.time()
    test_deterministic_query(train_dataloader, train_kg, 'set', None, ['((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))'])
    time2 = time.time()
    test_deterministic_query(train_dataloader, train_ckpt, 'vec', cuda_device, ['((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))'])
    time3 = time.time()
    test_deterministic_query(train_dataloader, train_kg, 'solver', None, ['((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))'])
    time4 = time.time()
    print(f"set time: {time2-time1}, FIT time: {time3 - time2}, solver time: {time4 - time3}")
"""



'''
train_dataloader = QueryAnsweringSeqDataLoader_v2(
        osp.join(data_folder, 'train-qaa.json'),
        # size_limit=args.batch_size * 1,
        target_lstr=['r1(s1,e1)&!r2(e1,f)&r3(s2,f)'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
train_dataloader = QueryAnsweringSeqDataLoader(
    osp.join(data_folder, 'valid-qaa.json'),
    # size_limit=args.batch_size * 1,
    target_lstr=train_queries,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
fof_list = train_dataloader.get_fof_list()
t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
for ifof, fof in t:
    print(fof)
'''