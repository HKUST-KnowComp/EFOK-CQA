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
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFOX-final')
parser.add_argument("--num_p", type=int, default=1000)
parser.add_argument("--num_n", type=int, default=500)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    all_formula_data = pd.read_csv(osp.join('data', 'DNF_EFO2_23_4123166.csv'))
    for i, row in tqdm.tqdm(all_formula_data.iterrows(), total=len(all_formula_data)):
        formula_id = row['formula_id']
        formula = row['formula']
        # data_path = osp.join(configure['data']['data_folder'], f'test_type{i:04d}_EFOX_qaa.json')
        data_path = osp.join(args.data_folder, f'test_{formula_id}_EFOX_qaa.json')
        if not osp.exists(data_path):
            print(f"{formula_id} not sampled")
            continue
        else:
            with open(data_path, 'rt') as f:
                old_data = json.load(f)
            new_data = {formula: []}
            f_str_list = [f'f{i + 1}' for i in range(row['f_num'])]
            f_str = '_'.join(f_str_list)
            if f_str not in old_data[formula][0][2]:
                print(f"warning, the {formula_id} need to be rectified")
                old_f_str = list(old_data[formula][0][2].keys())[0]
                for query_instance in old_data[formula]:
                    qa_dict, easy_ans_dict, hard_ans_dict = query_instance
                    new_data_instance = [qa_dict, {f_str: easy_ans_dict[old_f_str]}, {f_str: hard_ans_dict[old_f_str]}]
                    new_data[formula].append(new_data_instance)
                with open(data_path, 'wt') as f:
                    json.dump(new_data, f)
            else:
                if '!' not in formula:
                    assert len(old_data[formula]) >= args.num_p
                    if len(old_data[formula]) > args.num_p:
                        print(f"{formula_id} has {len(old_data[formula])} instances")
                        old_data[formula] = old_data[formula][:args.num_p]

                        with open(data_path, 'wt') as f:
                            json.dump(old_data, f)

                else:
                    assert len(old_data[formula]) >= args.num_n
                    if len(old_data[formula]) > args.num_n:
                        print(f"{formula_id} has {len(old_data[formula])} instances")
                        old_data[formula] = old_data[formula][:args.num_n]
                        with open(data_path, 'wt') as f:
                            json.dump(old_data, f)



