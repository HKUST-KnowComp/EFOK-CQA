import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import List

import torch
import pickle

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--input_folder", type=str, default='matrix/FB15k-237-Distmult')
parser.add_argument("--output_folder", type=str, default='sparse/FB15k-237-Distmult')
parser.add_argument("--threshold", type=int, default=0.05)
parser.add_argument("--epsilon", type=int, default=0.001)
parser.add_argument("--split", type=int, default=6)


def create_matrix_from_ckpt(scoring_matrix_list, observed_kg: KnowledgeGraph, real_starting_r, threshold, epsilon):
    n_rel, n_entity = len(scoring_matrix_list), scoring_matrix_list[0].shape[0]
    sparse_matrix_list = []
    for rel_id in range(n_rel):
        now_score_matrix = scoring_matrix_list[rel_id].to_dense()
        for h_id in range(n_entity):
            tail_set = observed_kg.hr2t[(h_id, rel_id + real_starting_r)]
            now_score_matrix[h_id] = torch.where(now_score_matrix[h_id] > threshold,
                                                 now_score_matrix[h_id], torch.zeros(n_entity))
            now_score_matrix[h_id] = now_score_matrix[h_id].clamp(0, 1 - epsilon)
            now_score_matrix[h_id][list(tail_set)] = 1
        sparse_matrix_list.append(now_score_matrix.to_sparse())
    return sparse_matrix_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'train_kg.tsv'),
        kgindex=kgidx)

    old_threshold, old_epsilon = 0.005, 0.001
    split_each_num = int(train_kg.num_relations / args.split)
    all_matrix_list = []
    for split_id in range(0, args.split):
        matrix_path = osp.join(args.input_folder, f'split_{split_id}_matrix_{old_threshold}_{old_epsilon}.ckpt')
        sparse_matrix_part = torch.load(matrix_path)
        real_starting_r = int(split_id * split_each_num)
        if old_threshold == args.threshold and old_epsilon == args.epsilon:
            filtered_m_list = sparse_matrix_part
        else:
            filtered_m_list = create_matrix_from_ckpt(sparse_matrix_part, train_kg, real_starting_r, args.threshold,
                                                      args.epsilon)
        all_matrix_list.extend(filtered_m_list)
        print(f'matrix of {split_id} finished')
    torch.save(all_matrix_list, osp.join(args.output_folder, f'torch_{args.threshold}_{args.epsilon}.ckpt'))
