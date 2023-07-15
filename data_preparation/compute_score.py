import argparse
import os.path as osp
from math import ceil

import torch

from data_preparation.create_matrix import create_matrix_from_ckpt
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default='cqd_models/FB15k-237.ckpt')
parser.add_argument("--ckpt_type", type=str, default='cqd', choices=['cqd', 'kge'])
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--cuda", type=int, default=2)
parser.add_argument("--split_num", type=int, default=6)
parser.add_argument("--batch", type=int, default=1000)
parser.add_argument("--output_folder", type=str, default='matrix/FB15k-237-Distmult')


def compute_batch_score_complex(rel, arg1, arg2, rank):
    rel_real, rel_img = rel[:, :, :rank], rel[:, :, rank:]
    arg1_real, arg1_img = arg1[:, :, :rank], arg1[:, :, rank:]
    arg2_real, arg2_img = arg2[:, :, :rank], arg2[:, :, rank:]

    # [B] Tensor
    score1 = torch.sum(rel_real * arg1_real * arg2_real, -1)
    score2 = torch.sum(rel_real * arg1_img * arg2_img, -1)
    score3 = torch.sum(rel_img * arg1_real * arg2_img, -1)
    score4 = torch.sum(rel_img * arg1_img * arg2_real, -1)
    res = score1 + score2 + score3 - score4
    del score1, score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img

    return res


def compute_batch_score_transe(rel, h_emb, t_emb):
    difference = rel + h_emb - t_emb
    score = -(torch.linalg.norm(difference, dim=-1))
    return score


def compute_batch_score_distmult(rel, h_emb, t_emb):
    score = torch.sum(rel * h_emb * t_emb, dim=-1)
    return score


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    data_folder = args.data_folder
    cqd_path = args.ckpt_path
    kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(data_folder, 'train_kg.tsv'),
        kgindex=kgidx)
    threshold, epsilon = 0.005, 0.001

    cqd_ckpt = torch.load(cqd_path)
    if args.ckpt_type == 'cqd':
        ent_emb = cqd_ckpt['embeddings.0.weight'].to(device)
        rel_emb = cqd_ckpt['embeddings.1.weight'].to(device)
    else:
        model_param = cqd_ckpt['model'][0]
        ent_emb = model_param['_entity_embedder.embeddings.weight']
        rel_emb = model_param['_relation_embedder.embeddings.weight']
    n_rel, n_ent = rel_emb.shape[0], ent_emb.shape[0]
    split_num = args.split_num
    split_each_relation = int(n_rel / split_num)
    batch_head = args.batch
    head_total_batch = ceil(n_ent / batch_head)
    for split in range(0, 3):
        sparse_list = []
        for relation_id in range(split_each_relation):
            all_matrix = torch.zeros((1, n_ent, n_ent), requires_grad=False)
            relation_total_id = relation_id + split * split_each_relation
            print('r_id', relation_total_id)
            for head_batch_id in range(head_total_batch):
                starting_h_id = int(head_batch_id * batch_head)
                batch_head_emb = ent_emb[starting_h_id: starting_h_id + batch_head, :]
                batch_head_emb = batch_head_emb.unsqueeze(-2)
                tail_emb = ent_emb.unsqueeze(0)
                this_rel_emb = rel_emb[relation_total_id].unsqueeze(0).unsqueeze(0)
                batch_score = compute_batch_score_distmult(this_rel_emb, batch_head_emb, tail_emb)
                all_matrix[0, starting_h_id: starting_h_id + batch_head] = batch_score
                del tail_emb, batch_score, batch_head_emb, this_rel_emb
            sparse_one_list = create_matrix_from_ckpt(all_matrix, train_kg, relation_total_id, threshold,
                                                      epsilon)
            del all_matrix
            sparse_list.extend(sparse_one_list)
        torch.save(sparse_list, osp.join(args.output_folder, f'split_{split}_matrix_{threshold}_{epsilon}.ckpt'))
        print(f"split{split} saved")
