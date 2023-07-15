import argparse
import json
import os.path as osp

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.language.grammar import parse_lstr_to_disjunctive_formula

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default='data/FB15k-EFO1')
parser.add_argument("--data_folder", type=str, default='data/FB15k-betae')
parser.add_argument("--mode", type=str, default='test')


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
    f_old = open(osp.join(args.output_folder, f'{args.mode}-qaa.json'))
    old_data = json.load(f_old)
    pni_data = old_data.pop('((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))')
    pni_instance = parse_lstr_to_disjunctive_formula('((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))')
    now_index = -1
    for i, query in enumerate(pni_data):
        qa_dict, easy_ans, hard_ans = query
        pni_instance.append_qa_instances(qa_dict)
        now_index += 1
        if args.mode == 'train':
            hard_answer = pni_instance.deterministic_query(now_index, train_kg)
            easy_answer = set()
        elif args.mode == 'valid':
            hard_answer = pni_instance.deterministic_query(now_index, valid_kg)
            easy_answer = pni_instance.deterministic_query(now_index, train_kg)
        else:
            hard_answer = pni_instance.deterministic_query(now_index, test_kg)
            easy_answer = pni_instance.deterministic_query(now_index, valid_kg)
        correct_query = [qa_dict, {'f': list(easy_answer)}, {'f': list(hard_answer - easy_answer)}]
        pni_data[i] = correct_query
        if i % 100 == 0:
            print(f'{i} has been finished.')
    new_data = json.load(open(osp.join(args.output_folder, f'{args.mode}_real_EFO1_qaa.json')))
    new_data['((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))'] = pni_data
    with open(osp.join(args.output_folder, f'{args.mode}_real_EFO1_qaa.json'), 'wt') as f:
        json.dump(new_data, f)


