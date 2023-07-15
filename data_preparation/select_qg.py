import os
import os.path as osp

import json
import pandas as pd
import tqdm
from shutil import copy

if __name__ == "__main__":
    old_data = pd.read_csv('data/DNF_EFO2_23_412316.csv')
    new_data = pd.read_csv('data/DNF_EFO2_23_4123166.csv')
    for dataset in ['FB15k-237', 'FB15k', 'NELL']:
        old_data_folder = osp.join('data', dataset + '-EFOX')
        new_data_folder = osp.join('data', dataset + '-EFOX-filtered')
        if not osp.exists(new_data_folder):
            os.makedirs(new_data_folder)
            copy(osp.join(old_data_folder, 'kgindex.json'), osp.join(new_data_folder, 'kgindex.json'))
            copy(osp.join(old_data_folder, 'train_kg.tsv'), osp.join(new_data_folder, 'train_kg.tsv'))
            copy(osp.join(old_data_folder, 'valid_kg.tsv'), osp.join(new_data_folder, 'valid_kg.tsv'))
            copy(osp.join(old_data_folder, 'test_kg.tsv'), osp.join(new_data_folder, 'test_kg.tsv'))
        for i, row in tqdm.tqdm(new_data.iterrows(), total=len(new_data)):
            formula = row['formula']
            formula_id = row['formula_id']
            if formula not in old_data['formula'].values:
                assert False, f'{formula} not in old data'
            else:
                old_formula_id = old_data[old_data['formula'] == formula]['formula_id'].values[0]
                if osp.exists(osp.join(old_data_folder, f'test_{old_formula_id}_EFOX_qaa.json')):
                    if not osp.exists(osp.join(new_data_folder, f'test_{formula_id}_EFOX_qaa.json')):
                        copy(osp.join(old_data_folder, f'test_{old_formula_id}_EFOX_qaa.json'),
                             osp.join(new_data_folder, f'test_{formula_id}_EFOX_qaa.json'))
                    else:
                        with open(osp.join(new_data_folder, f'test_{formula_id}_EFOX_qaa.json'), 'rt') as f:
                            old_data = json.load(f)
                            assert len(old_data) == 1
                            assert formula in old_data
                else:
                    print(f'{formula_id} not sampled')


