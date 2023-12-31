import argparse
import json
import logging
import os
import os.path as osp
import random
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm

from src.language.grammar import parse_lstr_to_lformula
from src.language.tnorm import Tnorm
from src.pipeline import (LMPNNReasoner, GradientEFOReasoner, LogicalMPLayer,
                          Reasoner)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader
from train_lmpnn import name2lstr, lstr2name, negation_query

torch.autograd.set_detect_anomaly(True)

from convert_beta_dataset import beta_lstr2name



parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-EFOX-final')
parser.add_argument("--train_queries",action='append')
parser.add_argument("--eval_queries", default="DNF_EFO2_23_4123166.csv",  action='append')
parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
parser.add_argument("--batch_size_eval_truth_value", type=int, default=32, help="batch size for evaluating the truth value")
parser.add_argument("--batch_size_eval_dataloader", type=int, default=5000, help="batch size for evaluation")

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--checkpoint_path", default="ckpt/FB15k-237/CQD/FB15k-237-model-rank-1000-epoch-100-1602508358.pt", type=str, help="path to the KGE checkpoint")
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=10)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--p", type=int, default=1)

# optimization for the entire process
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--noisy_sample_size", type=int, default=128)
parser.add_argument("--temp", type=float, default=0.05)

# reasoning machine
parser.add_argument("--reasoner", type=str, default='lmpnn', choices=['lmpnn', 'gradient', 'beam'])
parser.add_argument("--tnorm", type=str, default='product', choices=['product', 'godel'])

# reasoner = gradient
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_steps", type=int, default=1000)
parser.add_argument("--reasoning_optimizer", type=str, default='AdamW')

# reasoner = gnn
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--depth_shift", type=int, default=0)
parser.add_argument("--agg_func", type=str, default='sum')
parser.add_argument("--checkpoint_path_lmpnn", type=str, default="ckpt/FB15k-237/LMPNN/lmpnn-FB15K-237.ckpt")


def log_add_metric(add_log, mrr, h1, h3, h10, mul_mrr, h1_1, h3_3, h10_10):
    add_log['MRR'] += mrr
    add_log['HITS1'] += h1
    add_log['HITS3'] += h3
    add_log['HITS10'] += h10
    add_log['num_queries'] += 1
    add_log['couple_MRR'] += mul_mrr
    add_log['HITS1*1'] += h1_1
    add_log['HITS3*3'] += h3_3
    add_log['HITS10*10'] += h10_10
    return add_log


def ranking2metrics(ranking, easy_ans, hard_ans, ranking_device):
    num_hard = len(hard_ans)
    num_easy = len(easy_ans)
    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
    # only take those answers' rank
    cur_ranking = ranking[list(easy_ans) + list(hard_ans)]
    cur_ranking, indices = torch.sort(cur_ranking)
    masks = indices >= num_easy
    answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(ranking_device)
    cur_ranking = cur_ranking - answer_list + 1
    # filtered setting: +1 for start at 0, -answer_list for ignore other answers
    cur_ranking = cur_ranking[masks]
    # only take indices that belong to the hard answers
    mrr = torch.mean(1. / cur_ranking).item()
    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
    h10 = torch.mean(
        (cur_ranking <= 10).to(torch.float)).item()
    return mrr, h1, h3, h10


def evaluate_batch_joint(final_ranking, easy_ans_list, hard_ans_list, device, f_str):
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    for i in range(final_ranking.shape[0]):
        easy_ans = easy_ans_list[i][f_str]  # A list of list, each list is an instance.
        hard_ans = hard_ans_list[i][f_str]
        num_easy, num_hard = len(easy_ans), len(hard_ans)
        #  assert len(set(hard_ans).intersection(set(easy_ans))) == 0
        full_ans = easy_ans + hard_ans
        full_ans_tensor = torch.tensor(full_ans).to(device).transpose(0, 1)
        marginal_easy_ans_list, marginal_hard_ans_list = [], []
        couple_filtered_list = []
        marginal_exist_num = 0
        marginal_stored = []
        for j in range(final_ranking.shape[1]):
            marginal_easy_ans, marginal_full_ans = set([easy_instance[j] for easy_instance in easy_ans]), \
                set([full_instance[j] for full_instance in full_ans])
            marginal_hard_ans = marginal_full_ans - marginal_easy_ans
            marginal_easy_ans_list.append(marginal_easy_ans)
            marginal_hard_ans_list.append(marginal_hard_ans)
            marginal_ans_ranking = final_ranking[i, j][list(marginal_easy_ans) + list(marginal_hard_ans)]
            marginal_num_hard, marginal_num_easy = len(marginal_hard_ans), len(marginal_easy_ans)
            sort_marginal_ranking, marginal_indices = torch.sort(marginal_ans_ranking)
            marginal_masks = marginal_indices >= marginal_num_easy
            marginal_answer_list = torch.arange(
                marginal_num_hard + marginal_num_easy).to(torch.float).to(device)
            filtered_marginal_ranking = sort_marginal_ranking - marginal_answer_list + 1
            # filtered setting: +1 for start at 0, -answer_list for ignore other answers
            adjusted_marginal_all_ranking = final_ranking[i, j].clone().to(device)
            adjusted_marginal_all_ranking[list(marginal_easy_ans) + list(marginal_hard_ans)] = \
                torch.gather(filtered_marginal_ranking, dim=0, index=marginal_indices.argsort()).to(adjusted_marginal_all_ranking.dtype)
            couple_filtered_list.append(adjusted_marginal_all_ranking)
            #  Compute the marginal ranking first
            if len(marginal_hard_ans) == 0:  # There is really possibility that no marginal hard answer
                pass
            else:
                marginal_exist_num += 1
                marginal_hard_ranking = filtered_marginal_ranking[marginal_masks]
                marginal_mrr = torch.mean(1. / marginal_hard_ranking).item()
                marginal_h1 = torch.mean((marginal_hard_ranking <= 1).to(torch.float)).item()
                marginal_h3 = torch.mean((marginal_hard_ranking <= 3).to(torch.float)).item()
                marginal_h10 = torch.mean(
                    (marginal_hard_ranking <= 10).to(torch.float)).item()
                marginal_stored.append([marginal_mrr, marginal_h1, marginal_h3, marginal_h10])
        marginal_filtered_joint_rank = torch.stack(couple_filtered_list, dim=0)  # free_num * nentity
        m_j_ranking = torch.gather(
            marginal_filtered_joint_rank, dim=1, index=torch.tensor(hard_ans).to(device).transpose(0, 1))
        #  free_num * hard_num

        couple_mrr = torch.mean(torch.sqrt(torch.prod((1. / m_j_ranking), dim=0))).item()
        couple_h1 = torch.mean(torch.prod((m_j_ranking <= 1).to(torch.float), dim=0)).item()
        couple_h3 = torch.mean(torch.prod((m_j_ranking <= 3).to(torch.float), dim=0)).item()
        couple_h10 = torch.mean(torch.prod((m_j_ranking <= 10).to(torch.float), dim=0)).item()

        #  Compute the hard joint ranking
        couple_ans_ranking = torch.gather(final_ranking[i], dim=1, index=full_ans_tensor)  # free_num * ans
        add_ans_ranking = torch.sum(couple_ans_ranking, dim=0)  # ans
        final_ans_ranking = add_ans_ranking * (add_ans_ranking + 1) / 2 + couple_ans_ranking[0]
        sort_ans_ranking, indices = torch.sort(final_ans_ranking)
        masks = indices >= num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(device)
        filtered_ans_ranking = sort_ans_ranking - answer_list + 1
        cur_ranking = filtered_ans_ranking[masks]
        # if math.isinf(mrr):
        # print("warning: mrr is inf")
        mrr = torch.mean(1. / cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (cur_ranking <= 10).to(torch.float)).item()
        if marginal_exist_num == 0:
            no_marginal_logs = log_add_metric(
                no_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
        elif marginal_exist_num == 1:
            one_marginal_logs = log_add_metric(
                one_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
            one_marginal_logs['marginal_MRR'] += marginal_stored[0][0]
            one_marginal_logs['marginal_HITS1'] += marginal_stored[0][1]
            one_marginal_logs['marginal_HITS3'] += marginal_stored[0][2]
            one_marginal_logs['marginal_HITS10'] += marginal_stored[0][3]
        else:
            two_marginal_logs = log_add_metric(
                two_marginal_logs, mrr, h1, h3, h10, couple_mrr, couple_h1, couple_h3, couple_h10)
            two_marginal_logs['marginal_MRR'] += marginal_stored[0][0] / 2
            two_marginal_logs['marginal_HITS1'] += marginal_stored[1][0] / 2
            two_marginal_logs['marginal_HITS1'] += marginal_stored[0][1] / 2
            two_marginal_logs['marginal_HITS3'] += marginal_stored[1][1] / 2
            two_marginal_logs['marginal_HITS3'] += marginal_stored[0][2] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[1][2] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[0][3] / 2
            two_marginal_logs['marginal_HITS10'] += marginal_stored[1][3] / 2
    return two_marginal_logs, one_marginal_logs, no_marginal_logs


def eval_batch_query(model, pred_emb_list, easy_ans_list, hard_ans_list):
    """
    eval a batch of query of the same formula, the pred_emb of the query has been given.
    pred_emb:  batch*emb_dim
    easy_ans_list: list of easy_ans
    """
    device = model.device
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    f_str_list = [f'f{i + 1}' for i in range(len(pred_emb_list))]
    f_str = '_'.join(f_str_list)
    if len(pred_emb_list) == 1:
        with torch.no_grad():
            all_logit = model.compute_all_entity_logit(pred_emb_list[0], union=False)
            # batch*nentity
            argsort = torch.argsort(all_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            #  create a new torch Tensor for batch_entity_range
            ranking = ranking.scatter_(1, argsort, torch.arange(model.n_entity).to(torch.float).
                                       repeat(argsort.shape[0], 1).to(device))
            # achieve the ranking of all entities
            for i in range(all_logit.shape[0]):
                easy_ans = [instance[0] for instance in easy_ans_list[i][f_str]]
                hard_ans = [instance[0] for instance in hard_ans_list[i][f_str]]
                mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, device)
                two_marginal_logs['MRR'] += mrr
                two_marginal_logs['HITS1'] += h1
                two_marginal_logs['HITS3'] += h3
                two_marginal_logs['HITS10'] += h10
            num_query = all_logit.shape[0]
            two_marginal_logs['num_queries'] += num_query
    else:
        with torch.no_grad():
            final_ranking_list = []
            for pred_emb in pred_emb_list:
                all_logit = model.compute_all_entity_logit(pred_emb, union=False)
                argsort = torch.argsort(all_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                ranking = ranking.scatter_(1, argsort, torch.arange(model.n_entity).to(torch.float).
                                           repeat(argsort.shape[0], 1).to(device))
                final_ranking_list.append(ranking)
            final_ranking = torch.stack(final_ranking_list, dim=1).to(device)  # batch * free_num * nentity
            two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(final_ranking, easy_ans_list,
                                                                                          hard_ans_list, device, f_str)
    return two_marginal_logs, one_marginal_logs, no_marginal_logs


def train_LMPNN(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    T = args.temp
    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    nbp.eval()

    # for each batch
    for ifof, fof in t:
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_query(fof)

        # this procedure is somewhat of low efficiency
        reasoner.estimate_variable_embeddings()
        batch_fvar_emb = reasoner.get_ent_emb('f')
        pos_1answer_list = []
        neg_answers_list = []


        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        contrastive_pos_score = torch.exp(torch.cosine_similarity(
            batch_pos_emb, batch_fvar_emb, dim=-1) / T)
        contrastive_neg_score = torch.exp(torch.cosine_similarity(
            batch_neg_emb, batch_fvar_emb, dim=-1) / T)
        contrastive_nll = - torch.log(
            contrastive_pos_score / (contrastive_pos_score + contrastive_neg_score.sum(0))
        ).mean()


        metric_step['contrastive_pos_score'] = contrastive_pos_score.mean().item()
        metric_step['contrastive_neg_score'] = contrastive_neg_score.mean().item()
        metric_step['contrastive_nll'] = contrastive_nll.item()
        loss += contrastive_nll


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train LMPNN {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f1'
    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
        ranking = ranking.squeeze()
        if fof.hard_answer_list[i]:
            # [1, num_entities]
            hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                        device=nbp.device)
            hard_answer_rank = ranking[hard_answers]

            # remove better easy answers from its rankings
            if fof.easy_answer_list[i][k]:
                easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                            device=nbp.device)
                easy_answer_rank = ranking[easy_answers].view(-1, 1)

                num_skipped_answers = torch.sum(
                    hard_answer_rank > easy_answer_rank, dim=0)
                pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
            else:
                pure_hard_ans_rank = hard_answer_rank.squeeze()

        else:
            pure_hard_ans_rank = ranking[
                torch.tensor(fof.easy_answer_list[i][k], device=nbp.device)]

        # remove better hard answers from its ranking
        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
        num_skipped_answers = torch.sum(
            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
        )
        pure_hard_ans_rank -= num_skipped_answers.reshape(
            pure_hard_ans_rank.shape)

        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().float().numpy()
        hit1 = (pure_hard_ans_rank < 1).detach().cpu().float().numpy()
        hit3 = (pure_hard_ans_rank < 3).detach().cpu().float().numpy()
        hit10 = (pure_hard_ans_rank < 10).detach().cpu().float().numpy()
        metric['mrr'].append(rr.mean())
        metric['hit1'].append(hit1.mean())
        metric['hit3'].append(hit3.mean())
        metric['hit10'].append(hit10.mean())


def evaluate_by_search_emb_then_rank_truth_value(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    formula = list(dataloader.lstr_iterator.keys())[0]
    num_var = (1 if "f1" in formula else 0) + (1 if "f2" in formula else 0)
    f_str_list = [f'f{i + 1}' for i in range(num_var)]
    f_str = '_'.join(f_str_list)
    foqs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(foqs, desc=desc) as t:
        for query in t:
            reasoner.initialize_with_query(query)
            reasoner.estimate_variable_embeddings()
            if len(f_str_list) == 1:
                with torch.no_grad():
                    truth_value_entity_batch = reasoner.evaluate_truth_values(
                        free_var_emb_dict={
                            'f1': nbp.entity_embedding.unsqueeze(1),
    #                        "f2": reasoner.term_local_emb_dict["f2"]
                        },
                        batch_size_eval=args.batch_size_eval_truth_value)  # [num_entities batch_size]
                ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
                ranked_entity_ids = torch.argsort(
                    ranking_score, dim=-1, descending=True)
                batch_entity_rankings = torch.argsort(
                    ranked_entity_ids, dim=-1, descending=False)
                ranking = batch_entity_rankings
                for i in range(ranking.shape[0]):
                    easy_ans = [instance[0] for instance in query.easy_answer_list[i][f_str]]
                    hard_ans = [instance[0] for instance in query.hard_answer_list[i][f_str]]
                    mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, nbp.device)
                    two_marginal_logs['MRR'] += mrr
                    two_marginal_logs['HITS1'] += h1
                    two_marginal_logs['HITS3'] += h3
                    two_marginal_logs['HITS10'] += h10
                    two_marginal_logs["num_queries"] = ranking.shape[0]
            else:
                final_ranking_list = []
                for i in range(len(f_str_list)):
                    with torch.no_grad():
                        truth_value_entity_batch = reasoner.evaluate_truth_values(
                            free_var_emb_dict={
                                f'f{i+1}': nbp.entity_embedding.unsqueeze(1),
        #                        "f2": reasoner.term_local_emb_dict["f2"]
                            },
                            batch_size_eval=args.batch_size_eval_truth_value)  # [num_entities batch_size]
                    ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
                    ranked_entity_ids = torch.argsort(
                        ranking_score, dim=-1, descending=True)
                    batch_entity_rankings = torch.argsort(
                        ranked_entity_ids, dim=-1, descending=False)
                    final_ranking_list.append(batch_entity_rankings)
                final_ranking = torch.stack(final_ranking_list, dim=1).to(nbp.device)  # batch * free_num * nentity
                two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(final_ranking, query.easy_answer_list,
                                                                                          query.hard_answer_list, nbp.device, f_str)
    return two_marginal_logs, one_marginal_logs, no_marginal_logs



def evaluate_by_nearest_search(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: GradientEFOReasoner):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    two_marginal_logs = defaultdict(float)
    one_marginal_logs, no_marginal_logs = defaultdict(float), defaultdict(float)
    fofs = dataloader.get_fof_list()
    formula = list(dataloader.lstr_iterator.keys())[0]
    num_var = (1 if "f1" in formula else 0) + (1 if "f2" in formula else 0)
    f_str_list = [f'f{i + 1}' for i in range(num_var)]
    f_str = '_'.join(f_str_list)
    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                if num_var == 1:
                    ranking_list = []
                    reasoner.initialize_with_query(fof)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb = reasoner.get_ent_emb('f1')
                    batch_entity_rankings = nbp.get_all_entity_rankings(
                        batch_fvar_emb, score="cos")
                    ranking_list.append(batch_entity_rankings)

                elif num_var == 2:
                    reasoner.initialize_with_query(fof)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb_list = [[], []]
                    if "f1" in formula:
                        batch_f1_emb = reasoner.get_ent_emb('f1')
                        batch_fvar_emb_list[0].append(nbp.get_all_entity_rankings(
                            batch_f1_emb, score="cos"))
                    if "f2" in formula:
                        batch_f2_emb = reasoner.get_ent_emb('f2')
                        batch_fvar_emb_list[1].append(nbp.get_all_entity_rankings(
                            batch_f2_emb, score="cos"))
        if num_var == 1:
            ranking = torch.cat(ranking_list, dim=1)
            #  create a new torch Tensor for batch_entity_range
            # achieve the ranking of all entities
            for i in range(ranking.shape[0]):
                easy_ans = [instance[0] for instance in fof.easy_answer_list[i][f_str]]
                hard_ans = [instance[0] for instance in fof.hard_answer_list[i][f_str]]
                mrr, h1, h3, h10 = ranking2metrics(ranking[i], easy_ans, hard_ans, nbp.device)
                two_marginal_logs['MRR'] += mrr
                two_marginal_logs['HITS1'] += h1
                two_marginal_logs['HITS3'] += h3
                two_marginal_logs['HITS10'] += h10
                two_marginal_logs["num_queries"] = ranking.shape[0]
        elif num_var == 2:
            final_ranking_list = [torch.cat(ranking_list, dim=1) for ranking_list in batch_fvar_emb_list]
            final_ranking = torch.stack(final_ranking_list, dim=1).to(nbp.device)
            two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_batch_joint(final_ranking, fof.easy_answer_list,
                                                                                          fof.hard_answer_list, nbp.device, f_str)


    return two_marginal_logs, one_marginal_logs, no_marginal_logs



if __name__ == "__main__":
    # * parse argument
    args = parser.parse_args()

    # * prepare the logger
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    # * initialize the kgindex
    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    # * load neural binary predicate
    print(f"loading the nbp {args.model_name}")
    nbp = get_nbp_class(args.model_name)(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        scale=args.scale,
        device=args.device)


    if args.checkpoint_path:
        print("loading model from", args.checkpoint_path)
        nbp.load_state_dict(torch.load(args.checkpoint_path), strict=True)

    nbp.to(args.device)
    print(f"model loaded from {args.checkpoint_path}")

    # * load the dataset, by default, we load the dataset to test
    print("loading dataset")
    info_eval_queries = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    if args.eval_queries:
        eval_queries = list(info_eval_queries.formula)
    else:
        eval_queries = list(name2lstr.values())
    print("eval queries", eval_queries)

#    valid_dataloader = QueryAnsweringSeqDataLoader(
#        osp.join(args.task_folder, 'valid-qaa.json'),
#        target_lstr=eval_queries,
#        batch_size=args.batch_size_eval_dataloader,
#        shuffle=False,
#        num_workers=0)


    if args.reasoner in ['gradient', 'beam']:
        # for those reasoners without training
        tnorm = Tnorm.get_tnorm(args.tnorm)

        # todo: add more reasoners
        reasoner = GradientEFOReasoner(
            nbp,
            tnorm,
            reasoning_rate=args.reasoning_rate,
            reasoning_steps=args.reasoning_steps,
            reasoning_optimizer=args.reasoning_optimizer)

        all_log = defaultdict(dict)
        for i in range(len(eval_queries)):
            test_dataloader = QueryAnsweringSeqDataLoader(
                        osp.join(args.task_folder, 'test_type{:0>4d}_EFOX_qaa.json'.format(i)),
                        target_lstr=eval_queries,
                        batch_size=args.batch_size_eval_dataloader,
                        shuffle=False,
                        num_workers=0)
            two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_by_search_emb_then_rank_truth_value(
                        -1, f"evaluate test set", test_dataloader, nbp, reasoner)
            log = {eval_queries[i] : [two_marginal_logs, one_marginal_logs, no_marginal_logs]}
            all_log[eval_queries[i]] = log
            if osp.exists(
                    'EFO-1_log/{}_result/cqd_test'.format(args.task_folder.split("/")[-1].split("-")[0])) == False:
                os.makedirs('EFO-1_log/{}_result/cqd_test'.format(args.task_folder.split("/")[-1].split("-")[0]))
            with open('EFO-1_log/{}_result/cqd_test/all_logging_test_0_type{:0>4d}.pickle'.format(args.task_folder.split("/")[-1].split("-")[0], i), 'wb') as f:
                                pickle.dump(log, f)
        with open('EFO-1_log/{}_result/cqd_test/all_log.pickle'.format(args.task_folder.split("/")[-1].split("-")[0]), 'wb') as g:
                    pickle.dump(all_log, g)

    elif args.reasoner == 'lmpnn':
        # for those reasoners with training
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

#        train_dataloader = QueryAnsweringSeqDataLoader(
#            osp.join(args.task_folder, 'train-qaa.json'),
#            target_lstr=train_queries,
#            batch_size=args.batch_size,
#            shuffle=True,
#            num_workers=0)
        lgnn_layer = LogicalMPLayer(hidden_dim=args.hidden_dim,
                                     nbp=nbp,
                                     layers=args.num_layers,
                                     eps=args.eps,
                                     agg_func=args.agg_func)

        if args.checkpoint_path_lmpnn:
            print("loading lmpnn model from", args.checkpoint_path_lmpnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_lmpnn, map_location=args.device), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = LMPNNReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        if args.train_queries:
            train_dataloader, valid_dataloader, test_dataloader = None, None, None
            for e in range(1, 1+args.epoch):
                train_LMPNN(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
                scheduler.step()
                if e % 5 == 0:
                    evaluate_by_nearest_search(e, f"NN evaluate validate set epoch {e}",
                                            valid_dataloader, nbp, reasoner)
                    evaluate_by_nearest_search(e, f"NN evaluate test set epoch {e}",
                                            test_dataloader, nbp, reasoner)

                    save_name = os.path.join(args.output_dir,
                                            f'lmpnn-{e}.ckpt')
                    torch.save(lgnn_layer.state_dict(), save_name)

                    last_name = os.path.join(args.output_dir,
                                            f'lmpnn-last.ckpt')
                    torch.save(lgnn_layer.state_dict(), last_name)

            if args.epoch == 0:
                evaluate_by_nearest_search(e, f"NN evaluate validate set",
                                            valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search(e, f"NN evaluate test set ",
                                            test_dataloader, nbp, reasoner)
        else:
            all_log = defaultdict(dict)
            for i in range(len(eval_queries)):
                test_dataloader = QueryAnsweringSeqDataLoader(
                    osp.join(args.task_folder, 'test_type{:0>4d}_EFOX_qaa.json'.format(i)),
                    target_lstr=eval_queries,
                    batch_size=args.batch_size_eval_dataloader,
                    shuffle=False,
                    num_workers=0)
                two_marginal_logs, one_marginal_logs, no_marginal_logs = evaluate_by_nearest_search(0, f"NN evaluate test set ",
                                            test_dataloader, nbp, reasoner)
                log = {eval_queries[i] : [two_marginal_logs, one_marginal_logs, no_marginal_logs]}
                all_log[eval_queries[i]] = log
                if osp.exists('EFO-1_log/{}_result/lmpnn_test'.format(args.task_folder.split("/")[-1].split("-")[0])) == False:
                    os.makedirs('EFO-1_log/{}_result/lmpnn_test'.format(args.task_folder.split("/")[-1].split("-")[0]))
                with open('EFO-1_log/{}_result/lmpnn_test/all_logging_test_0_type{:0>4d}.pickle'.format(args.task_folder.split("/")[-1].split("-")[0], i), 'wb') as f:
                        pickle.dump(log, f)
            with open('EFO-1_log/{}_result/lmpnn_test/all_logging.pickle'.format(args.task_folder.split("/")[-1].split("-")[0]), 'wb') as g:
                    pickle.dump(all_log, g)
