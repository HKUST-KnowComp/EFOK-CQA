import argparse
import math
import os.path as osp
from collections import defaultdict

import torch
import tqdm
import json
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='data/FB15k-EFO1')
parser.add_argument("--output_folder", type=str, default='data/FB15k-EFOX-final')

formula_correspondence = {
    'r1(s1,f)': 'r1(s1,f1)',
    '(r1(s1,e1))&(r2(e1,f))': '(r1(s1,e1))&(r2(e1,f1))',
    'r1(s1,e1)&r2(e1,f)': '(r1(s1,e1))&(r2(e1,f1))',
    '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '(r1(s1,e1))&((r2(e1,e2))&(r3(e2,f1)))',
    'r1(s1,e1)&r2(e1,e2)&r3(e2,f)': '(r1(s1,e1))&((r2(e1,e2))&(r3(e2,f1)))',
    '(r1(s1,f))&(r2(s2,f))': '(r1(s1,f1))&(r2(s2,f1))',
    'r1(s1,f)&r2(s2,f)': '(r1(s1,f1))&(r2(s2,f1))',
    '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '(r1(s1,f1))&((r2(s2,f1))&(r3(s3,f1)))',
    'r1(s1,f)&r2(s2,f)&r3(s3,f)': '(r1(s1,f1))&((r2(s2,f1))&(r3(s3,f1)))',
    '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': '(r1(s1,e1))&((r2(s2,e1))&(r3(e1,f1)))',
    'r1(s1,e1)&r2(s2,e1)&r3(e1,f)': '(r1(s1,e1))&((r2(s2,e1))&(r3(e1,f1)))',
    '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f1))',
    'r1(s1,e1)&r2(e1,f)&r3(s2,f)': '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f1))',
    '(r1(s1,f))&(!(r2(s2,f)))': '(r1(s1,f1))&(!(r2(s2,f1)))',
    'r1(s1,f)&!r2(s2,f)': '(r1(s1,f1))&(!(r2(s2,f1)))',
    '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '((r1(s1,f1))&(r2(s2,f1)))&(!(r3(s3,f1)))',
    'r1(s1,f)&r2(s2,f)&!r3(s3,f)': '((r1(s1,f1))&(r2(s2,f1)))&(!(r3(s3,f1)))',
    '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))',
    'r1(s1,e1)&!r2(s2,e1)&r3(e1,f)': '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))',
    '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': '((r1(s1,e1))&(r2(e1,f1)))&(!(r3(s2,f1)))',
    'r1(s1,e1)&r2(e1,f)&!r3(s2,f)': '((r1(s1,e1))&(r2(e1,f1)))&(!(r3(s2,f1)))',
    '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': '((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))',
    'r1(s1,e1)&!r2(e1,f)&r3(s2,f)': '((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))',
    '(r1(s1,f))|(r2(s2,f))': '(r1(s1,f1))|(r2(s2,f1))',
    'r1(s1,f)|r2(s2,f)': '(r1(s1,f1))|(r2(s2,f1))',
    '((r1(s1,e1))&(r3(e1,f)))|((r2(s2,e1))&(r3(e1,f)))': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    '(r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    'r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))',
    '!(!r1(s1,f)&!r2(s2,f))': '(r1(s1,f1))|(r2(s2,f1))',
    '!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)': '((r1(s1,e1))&(r3(e1,f1)))|((r2(s2,e1))&(r3(e1,f1)))'
}

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    for mode in ['train', 'valid', 'test']:
        old_data_file = osp.join(args.data_folder, f'{mode}-qaa.json')
        new_data_file = osp.join(args.output_folder, f'{mode}-qaa.json')
        with open(old_data_file, 'rt') as f:
            old_data = json.load(f)
        new_data = defaultdict(list)
        for formula in old_data:
            new_formula = formula_correspondence[formula]
            if old_data[formula]:
                for query in old_data[formula]:
                    qa_dict, easy_ans_dict, hard_ans_dict = query
                    new_easy_ans_dict = {'f1': [[easy_ans] for easy_ans in easy_ans_dict['f']]}
                    if mode in ['train']:
                        new_hard_ans_dict = []
                    else:
                        new_hard_ans_dict = {'f1': [[hard_ans] for hard_ans in hard_ans_dict['f']]}
                    new_data[new_formula].append([qa_dict, new_easy_ans_dict, new_hard_ans_dict])
        with open(new_data_file, 'wt') as f:
            json.dump(new_data, f)

