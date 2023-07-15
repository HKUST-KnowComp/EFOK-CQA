import argparse
import os.path as osp
from collections import defaultdict

import torch
import tqdm

from FIT import solve_EFO1
from src.utils.data import QueryAnsweringSeqDataLoader_v2
from src.utils.class_util import Writer

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--ckpt", type=str, default='sparse/237/torch_0.005_0.001.ckpt')
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--mode", type=str, default='test', choices=['valid', 'test'])
parser.add_argument("--e_norm", type=str, default='Godel', choices=['Godel', 'product'])
parser.add_argument("--c_norm", type=str, default='product', choices=['Godel', 'product'])
parser.add_argument("--max", type=int, default=10)
parser.add_argument("--data_type", type=str, default='EFO1', choices=['BetaE', 'EFO1', 'EFO1_l'])
parser.add_argument("--formula", type=list, default=None)
negation_list = ['(r1(s1,f))&(!(r2(s2,f)))', '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))', '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))', '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))', '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))']


@torch.no_grad()
def compute_single_evaluation(fof, batch_ans_tensor, n_entity):
    k = 'f'
    metrics = defaultdict(float)
    argsort = torch.argsort(batch_ans_tensor, dim=1, descending=True)
    ranking = argsort.clone().to(torch.float).to(cuda_device)
    ranking = ranking.scatter_(1, argsort, torch.arange(n_entity).to(torch.float).
                               repeat(argsort.shape[0], 1).to(cuda_device))
    for i in range(batch_ans_tensor.shape[0]):
        #ranking = ranking.scatter_(0, argsort, torch.arange(n_entity).to(torch.float))
        hard_ans = fof.hard_answer_list[i][k]
        easy_ans = fof.easy_answer_list[i][k]
        num_hard = len(hard_ans)
        num_easy = len(easy_ans)
        real_ans_num = num_easy + num_hard
        pred_ans_num = torch.sum(batch_ans_tensor[i])
        cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
        cur_ranking, indices = torch.sort(cur_ranking)
        masks = indices >= num_easy
        # easy_masks = indices < num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).to(cuda_device)
        cur_ranking = cur_ranking - answer_list + 1
        # filtered setting: +1 for start at 0, -answer_list for ignore other answers
        # easy_ranking = cur_ranking[easy_masks]
        hard_ranking = cur_ranking[masks]
        # only take indices that belong to the hard answers
        '''
        if easy_ans:
            easy_mrr = torch.mean(1. / easy_ranking).item()
            metrics['easy_queries'] += 1
        else:
            easy_mrr = 0
        metrics['easy_MRR'] += easy_mrr
        '''
        mrr = torch.mean(1. / hard_ranking).item()
        h1 = torch.mean((hard_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((hard_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean(
            (hard_ranking <= 10).to(torch.float)).item()
        mae = torch.abs(pred_ans_num - real_ans_num).item()
        mape = mae / real_ans_num
        metrics['MAE'] += mae
        metrics['MAPE'] += mape
        metrics['MRR'] += mrr
        metrics['HITS1'] += h1
        metrics['HITS3'] += h3
        metrics['HITS10'] += h10
    metrics['num_queries'] += batch_ans_tensor.shape[0]
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    relation_matrix_list = torch.load(args.ckpt)
    n_relation, n_entity = len(relation_matrix_list), relation_matrix_list[0].shape[0]
    if args.cuda < 0:
        cuda_device = torch.device('cpu')
    else:
        cuda_device = torch.device('cuda:{}'.format(args.cuda))
    for i in range(len(relation_matrix_list)):
        relation_matrix_list[i] = relation_matrix_list[i].to(cuda_device)
    if args.data_type == 'BetaE':
        formula_path = osp.join(args.data_folder, f'{args.mode}-qaa.json')
    elif args.data_type == 'EFO1':
        formula_path = osp.join(args.data_folder, f'{args.mode}_real_EFO1_qaa.json')
    elif args.data_type == 'EFO1_l':
        formula_path = osp.join(args.data_folder, f'{args.mode}_1000_real_EFO1_qaa.json')
    else:
        raise NotImplementedError
    test_dataloader = QueryAnsweringSeqDataLoader_v2(
        formula_path,
        # size_limit=args.batch_size * 1,
        target_lstr=args.formula,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)
    writer = Writer(case_name=args.ckpt, config=args, log_path='results')
    fof_list = test_dataloader.get_fof_list_no_shuffle()
    t = tqdm.tqdm(enumerate(fof_list), total=len(fof_list))
    all_metrics = defaultdict(dict)
    # all_answers, now_formula_index = {}, {}
    # for lstr in test_dataloader.lstr_qaa:
        # all_answers[lstr] = torch.zeros((len(test_dataloader.lstr_qaa[lstr]), n_entity))
        # now_formula_index[lstr] = 0
    for ifof, fof in t:
        torch.cuda.empty_cache()
        batch_ans_list, metric = [], {}
        for query_index in range(len(fof.easy_answer_list)):
            ans = solve_EFO1(fof, relation_matrix_list, args.c_norm, args.e_norm, query_index, cuda_device, args.max)
            batch_ans_list.append(ans)
        batch_ans_tensor = torch.stack(batch_ans_list, dim=0)
        #all_answers[fof.lstr][now_formula_index[fof.lstr]: now_formula_index[fof.lstr] + batch_ans_tensor.shape[0], :] \
            #= batch_ans_tensor
        #now_formula_index[fof.lstr] += batch_ans_tensor.shape[0]
        batch_score = compute_single_evaluation(fof, batch_ans_tensor, n_entity)
        for metric in batch_score:
            if metric not in all_metrics[fof.lstr]:
                all_metrics[fof.lstr][metric] = 0
            all_metrics[fof.lstr][metric] += batch_score[metric]
        del batch_score, batch_ans_tensor
    for full_formula in all_metrics.keys():
        for log_metric in all_metrics[full_formula].keys():
            if log_metric != 'num_queries':
                all_metrics[full_formula][log_metric] /= all_metrics[full_formula]['num_queries']
    print(all_metrics)
    #writer.save_torch(all_answers, 'all_answer_tensor.ckpt')
    writer.save_pickle(all_metrics, f"all_logging_{args.mode}_0.pickle")
