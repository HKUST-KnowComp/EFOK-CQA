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
parser.add_argument("--ckpt", type=str, default='cqd_models')
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--action", choices=['prob', 'sparse', 'change'], default='prob')
parser.add_argument("--output_folder", type=str, default='sparse/237')
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--epsilon", type=float, default=0.001)
parser.add_argument("--split_num", type=int, default=79)


def create_matrix_statistics(scoring_matrix, observed_kg: KnowledgeGraph, latent_kg: KnowledgeGraph):
    n_rel, n_entity = scoring_matrix.shape[0], scoring_matrix.shape[1]
    only_hard_ans, easy_hard_ans, non_ans = [], [], []
    full_tail_prob = torch.softmax(scoring_matrix, dim=2)
    for rel_id in range(n_rel):
        for h_id in range(n_entity):
            tail_prob = full_tail_prob[rel_id][h_id]
            tail_set = observed_kg.hr2t[(h_id, rel_id)]
            hard_tail_set = latent_kg.hr2t[(h_id, rel_id)] - tail_set
            observed_t_num = len(tail_set)
            observed_edge_prob_vec = tail_prob[list(tail_set)]
            observed_edge_prob = torch.sum(observed_edge_prob_vec)
            hard_edge_prob_vec = tail_prob[list(hard_tail_set)]
            if tail_set and hard_tail_set:
                easy_hard_ans.append([observed_edge_prob_vec, hard_edge_prob_vec])
            if hard_tail_set and not tail_set:
                #max_hard_ans, least_hard_ans = min(hard_edge_prob_vec), max(hard_edge_prob_vec)
                only_hard_ans.append(hard_edge_prob_vec)
            if not hard_tail_set and not tail_set:
                max_non_answer = max(tail_prob)
                non_ans.append(max_non_answer)
    return only_hard_ans, easy_hard_ans, non_ans


def create_matrix_from_ckpt(scoring_matrix, observed_kg: KnowledgeGraph, real_starting_r, threshold=0.01, epsilon=0.01):
    n_rel, n_entity = scoring_matrix.shape[0], scoring_matrix.shape[1]
    full_tail_prob = torch.softmax(scoring_matrix, dim=2)
    sparse_matrix_list = []
    for rel_id in range(n_rel):
        for h_id in range(n_entity):
            tail_prob = full_tail_prob[rel_id][h_id]
            tail_set = observed_kg.hr2t[(h_id, rel_id + real_starting_r)]
            observed_t_num = len(tail_set)
            scailing = observed_t_num/torch.sum(tail_prob[list(tail_set)]) if observed_t_num else 1
            full_tail_prob[rel_id][h_id] *= scailing
            full_tail_prob[rel_id][h_id] = torch.where(full_tail_prob[rel_id][h_id] > threshold,
                                                       full_tail_prob[rel_id][h_id], torch.zeros(n_entity))
            full_tail_prob[rel_id][h_id] = full_tail_prob[rel_id][h_id].clamp(0, 1-epsilon)
            full_tail_prob[rel_id][h_id][list(tail_set)] = 1
        sparse_matrix_list.append(full_tail_prob[rel_id].to_sparse())
    return sparse_matrix_list


def collect_matrix_scipy_sparse(matrix_path):
    dense_matrix = torch.load(matrix_path)
    matrix_list = []
    for i in range(dense_matrix.shape[0]):
        sparse_matrix = dense_matrix[i]
        matrix_list.append(sparse_matrix)
    return matrix_list


def change_matrix_format(matrix_list):
    new_matrix_list = []
    for matrix in matrix_list:
        new_matrix_list.append(matrix.tocsc())
    return new_matrix_list


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kgidx = KGIndex.load(osp.join(args.data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    '''
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(args.data_folder, 'test_kg.tsv'),
        kgindex=kgidx)
    '''
    threshold, epsilon = args.threshold, args.epsilon
    split_num = args.split_num
    split_each_num = int(train_kg.num_relations / split_num)
    all_matrix_list = []
    for split_id in range(0, split_num):
        matrix_path = osp.join(args.output_folder, f'split_{split_id}_matrix_{threshold}_{epsilon}.ckpt')
        # whole_prob_matrix = torch.zeros(split_each_num, train_kg.num_entities, train_kg.num_entities)
        score_matrix = torch.load(osp.join(args.ckpt, f'matrix_{split_id}.ckpt'), map_location=None)
        real_starting_r = int(split_id * split_each_num)
        sparse_matrix_list = create_matrix_from_ckpt(score_matrix, train_kg, real_starting_r, threshold, epsilon)
        all_matrix_list.extend(sparse_matrix_list)
        torch.save(sparse_matrix_list, matrix_path)
        print(f'matrix of {split_id} finished')
    torch.save(all_matrix_list, osp.join(args.output_folder, f'torch_{args.threshold}_{args.epsilon}.ckpt'))

