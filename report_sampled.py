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



train_queries = list(name2lstr.values())
query_2in = 'r1(s1,f)&!r2(s2,f)'
query_2i = 'r1(s1,f)&r2(s2,f)'
parser = argparse.ArgumentParser()
#parser.add_argument("--output_name", type=str, default='new-qaa')
parser.add_argument("--dataset", type=str, default='FB15k')
parser.add_argument("--num_positive", type=int, default=800)
parser.add_argument("--num_negative", type=int, default=400)
parser.add_argument('--mode', choices=['train', 'valid', 'test'], default='test')
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=740)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    output_folder = osp.join('data', args.dataset + '-EFOX-filtered')
    formula_scope = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    all_sampled = 0
    for i, row in tqdm.tqdm(formula_scope.iterrows(), total=len(formula_scope)):
        if i > args.end_index or i < args.start_index:
            continue
        lstr = row.formula
        fid = row.formula_id
        output_file_name = osp.join(output_folder,
                                    f'{args.mode}_{fid}_EFOX_qaa.json')
        useful_num = 0
        all_qa_dict = set()
        if os.path.exists(output_file_name):
            with open(output_file_name, 'rt') as f:
                old_data = json.load(f)
            assert len(old_data) == 1
            assert lstr in old_data
            if '!' in lstr:
                if len(old_data[lstr]) >= args.num_negative:
                    all_sampled += 1
                else:
                    print(f'file {output_file_name} not enough negative samples, now {len(old_data[lstr])}')
            else:
                if len(old_data[lstr]) >= args.num_positive:
                    all_sampled += 1
                else:
                    print(f'file {output_file_name} not enough positive samples, now {len(old_data[lstr])}')
        else:
            print(f'file {output_file_name} not exist')
    print(f"During {args.start_index} {args.end_index} all_sampled", all_sampled, f"ratio is {all_sampled / (args.end_index - args.start_index + 1)}")
