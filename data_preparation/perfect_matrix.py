import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List

import pickle

import torch

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--output_folder", type=str, default='sparse/237')


def create_perfect_matrix(kg: KnowledgeGraph):
    entity_num, rel_num = kg.num_entities, kg.num_relations
    all_matrix_list = []
    for rel in kg.r2ht:
        new_matrix = torch.zeros((entity_num, entity_num), dtype=torch.int8)
        for head in range(entity_num):
            for tail in kg.hr2t[head, rel]:
                new_matrix[head, tail] = 1
        all_matrix_list.append(new_matrix.to_sparse())
    return all_matrix_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))

    mode_list = ['train', 'valid', 'test']
    for mode in mode_list:
        kg = KnowledgeGraph.create(
            triple_files=osp.join(args.data_folder, f'{mode}_kg.tsv'),
            kgindex=kgidx)
        matrix_path = osp.join(args.output_folder, f'torch_{mode}_perfect.ckpt')
        sparse_matrix_list = create_perfect_matrix(kg)
        torch.save(sparse_matrix_list, matrix_path)





    # torch.save(all_matrix_list, matrix_path)