import copy
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np
import sys
import shutil

from src.utils.class_util import fixed_depth_nested_dict



def fill_dict_to_array(whole_dict, array, depth2key, now_depth, full_depth):
    for index, key in enumerate(depth2key[now_depth]):
        if now_depth == full_depth:
            array[index] = whole_dict[key]
        else:
            fill_dict_to_array(whole_dict[key], array[index], depth2key, now_depth + 1, full_depth)


def nested_dict_to_array(nested_dictionary, meta_key_list):
    """
    key: p,e , MRR
    meta_key: formula/metric
    """
    nested_depth = len(meta_key_list)
    full_depth2key = {i: [] for i in range(nested_depth)}
    now_dict = nested_dictionary
    for depth in range(nested_depth):
        now_keys = now_dict.keys()
        full_depth2key[depth] = list(now_keys)
        now_dict = now_dict[full_depth2key[depth][0]]
    final_array = np.zeros([len(full_depth2key[i]) for i in range(nested_depth)])
    fill_dict_to_array(nested_dictionary, final_array, full_depth2key, 0, nested_depth - 1)
    return final_array, full_depth2key


def choose_test_wrt_valid(folder_path, valid_file, test_file):
    '''
    Use early stopping, namely choose the best score of testing wrt. the valid performance.
    '''


def remove_checkpoint(folder_path, exempt_step_list, exempt_largest: bool = True):
    print(folder_path)
    if exempt_step_list is None:
        exempt_step_list = []
    exempt_step_list.sort()
    exist_sub_dir = False
    for sub_file in os.listdir(folder_path):
        full_sub_path = os.path.join(folder_path, sub_file)
        if os.path.isdir(full_sub_path) and sub_file != '.ipynb_checkpoints':
            exist_sub_dir = True
            remove_checkpoint(full_sub_path, copy.deepcopy(exempt_step_list), exempt_largest)
    if not exist_sub_dir:
        file_list = os.listdir(folder_path)
        ckpt_step = [int(ckpt_file.split('.ck')[0])
                     for ckpt_file in file_list if ckpt_file.endswith('.ckpt')]
        ckpt_step.sort()
        if exempt_largest and ckpt_step and ckpt_step[-1] not in exempt_step_list:
            exempt_step_list.append(ckpt_step[-1])
        # print(ckpt_step, exempt_step_list, folder_path)
        for i, step in enumerate(ckpt_step):
            if step not in exempt_step_list:
                file_path = os.path.join(folder_path, f'{step}.ckpt')
                os.remove(file_path)
                print(f'Delete {step}.ckpt')


def merge_continue_folder(original_folder, continue_folder, ckpt_step_list, saving_step_list, saving_mode_list,
                          do_assertion: bool = True):
    ckpt_step_list.sort()
    saving_step_list.sort()
    shutil.copy(os.path.join(continue_folder, 'meta.json'), os.path.join(original_folder, 'continue_meta.json'))
    train_data = pd.read_csv(os.path.join(continue_folder, 'train.csv'))
    original_train_data = pd.read_csv(os.path.join(original_folder, 'train.csv'))
    original_train_steps = list(original_train_data['step'])
    with open(os.path.join(original_folder, 'train.csv'), 'at') as f:
        for index in train_data.index:
            step = train_data['step'].loc[index]
            if step in original_train_steps:
                if do_assertion:
                    assert original_train_data.loc[original_train_data['step'] == step].equals(
                        train_data.loc[train_data['step'] == step]), print(step)
            else:
                f.write(','.join([str(train_data[c].loc[index]) for c in train_data.columns]) + '\n')
    for step in ckpt_step_list:
        shutil.copy(os.path.join(continue_folder, f'{step}.ckpt'), os.path.join(original_folder, f'{step}.ckpt'))
    for step in saving_step_list:
        for mode in saving_mode_list:
            shutil.copy(os.path.join(continue_folder, f'all_logging_{mode}_{step}.pickle'),
                        os.path.join(original_folder, f'all_logging_{mode}_{step}.pickle'))


def new_merge_pickle(folder_path, step_list, meta_key_list, mode):
    all_logging = {}
    for step_id in range(len(step_list)):
        step = step_list[step_id]
        filename = f'all_logging_{mode}_{step}.pickle'
        with open(os.path.join(folder_path, filename), 'rb') as f:
            single_log = pickle.load(f)
            all_logging[step] = single_log
    final_array, depth2key = nested_dict_to_array(all_logging, meta_key_list)
    with open(os.path.join(folder_path, f'new_merge_logging_{mode}.pickle'), 'wb') as f:
        pickle.dump([final_array, depth2key, meta_key_list], f)


def new_read_merge_pickle(folder_path, fixed_dict, mode, transpose=False, percentage=False):
    with open(os.path.join(folder_path, f'new_merge_logging_{mode}.pickle'), 'rb') as f:
        single_log = pickle.load(f)
        final_array, depth2key, meta_key_list = single_log
        if percentage:
            final_array *= 100
        selected_index_list = []
        left_meta_key_index_list = []
        for i, meta_key in enumerate(meta_key_list):
            if meta_key in fixed_dict:
                index2key = depth2key[i]
                key2index = {index2key[j]: j for j in range(len(index2key))}
                fixed_index = key2index[fixed_dict[meta_key]]
            else:
                fixed_index = slice(len(depth2key[i]))
                left_meta_key_index_list.append(i)
            selected_index_list.append(fixed_index)
        assert len(left_meta_key_index_list) <= 2, "Get more than two meta keys unfixed!"
        selected_log = final_array[tuple(selected_index_list)]
        if len(left_meta_key_index_list) == 1:
            left_meta_key_index = left_meta_key_index_list[0]
            left_meta_key_name = meta_key_list[left_meta_key_index]
            reindexed_data = pd.DataFrame(data=selected_log, index=depth2key[left_meta_key_index_list[0]],
                                          columns=[str(fixed_dict)])
            reindexed_data.to_csv(os.path.join(folder_path, f'selected_log_{mode}_{left_meta_key_name}.csv'))
        elif len(left_meta_key_index_list) == 2:
            reindexed_data = pd.DataFrame(
                data=selected_log, index=depth2key[left_meta_key_index_list[0]],
                columns=depth2key[left_meta_key_index_list[1]])
            left_meta_key_name_list = [meta_key_list[i] for i in left_meta_key_index_list]
            if transpose:
                reindexed_data = reindexed_data.transpose()
            reindexed_data.to_csv(
                os.path.join(folder_path, f'selected_log_{mode}_'
                                          f'{left_meta_key_name_list[0]}_{left_meta_key_name_list[1]}.csv'))
        return reindexed_data


def pickle_select_form(pickle_path, test_step, meta_key_list, fixed_dict, normal_form, formula_file=None):
    if formula_file:
        formula_data = pd.read_csv(formula_file)
    else:
        formula_data = None
    new_merge_pickle(pickle_path, [test_step], meta_key_list, 'test')
    loading_data = new_read_merge_pickle(pickle_path, fixed_dict, 'test', False, False)
    if normal_form == 'best':
        best_formula_list, best_score_list = [], []
        for type_str in formula_data.index:
            now_best_score, now_best_formula = 0, None
            for possible_formula in formula_data.loc[type_str]:
                if possible_formula in loading_data.index:
                    if loading_data.loc[possible_formula].values[0] > now_best_score:
                        now_best_score, now_best_formula = loading_data.loc[possible_formula].values[0], \
                                                           possible_formula
            best_formula_list.append(now_best_formula)
            best_score_list.append(now_best_score)
        output_data = pd.DataFrame(data={'formula': best_formula_list, 'score': best_score_list},
                                   index=formula_data.index)
    else:
        all_formulas = formula_data[normal_form]
        normal_form_index_list = [i for i in loading_data.index if loading_data in all_formulas]
        output_data = loading_data.loc[normal_form_index_list]
    output_data.to_csv(os.path.join(pickle_path, f'chose_form_{normal_form}.csv'))


def process_output_whole_folder(whole_folder, meta_key_list, auto_delete, mode, fixed_dict, percentage=False,
                                transpose=False):
    output_dict = {}
    exist_sub_dir = False
    delete_folder = False
    for sub_file in os.listdir(whole_folder):
        full_sub_path = os.path.join(whole_folder, sub_file)
        if os.path.isdir(full_sub_path) and sub_file != '.ipynb_checkpoints':
            exist_sub_dir = True
            sub_output_dict = process_output_whole_folder(full_sub_path, meta_key_list, auto_delete, mode, fixed_dict,
                                                          percentage, transpose)
            output_dict.update(sub_output_dict)
    if not exist_sub_dir:  # The final dir that contains output
        now_model_name = whole_folder.split('/')[-1].split('_')[0]
        file_list = os.listdir(whole_folder)
        ckpt_step_list = [int(ckpt_file.split('.')[0]) for ckpt_file in file_list if ckpt_file.endswith('.ckpt')
                          and ckpt_file.split('.')[0].isdigit()]
        logging_mode_step = [int(logging_file.split('.')[0].split('_')[-1])
                             for logging_file in file_list if logging_file.endswith('.pickle')
                             and logging_file.split('.')[0].split('_')[-2] == mode and
                             logging_file.split('.')[0].split('_')[-1].isdigit()]
        largest_ckpt_step = max(ckpt_step_list) if ckpt_step_list else 0
        print(f'processing folder {whole_folder}， {len(logging_mode_step)}')
        if len(logging_mode_step):
            new_merge_pickle(whole_folder, sorted(logging_mode_step), meta_key_list, mode)
            new_fixed_dict = copy.deepcopy(fixed_dict)
            if 'step' in fixed_dict and fixed_dict['step'] == 'last':
                new_fixed_dict['step'] = max(logging_mode_step)
            new_read_merge_pickle(whole_folder, new_fixed_dict, mode=mode, percentage=percentage, transpose=transpose)
        output_dict[whole_folder] = largest_ckpt_step
        logging_file_num = len([log_name for log_name in file_list if log_name.endswith('pickle')])
        if largest_ckpt_step == 0 and logging_file_num == 0:
            delete_folder = True
    if delete_folder and not exist_sub_dir and auto_delete:
        shutil.rmtree(whole_folder)
    return output_dict


def aggregate_test(folder_path, prefix, delete_segmented):
    output_folder = os.path.join(folder_path, f'{prefix}_aggregated')
    os.makedirs(output_folder, exist_ok=True)
    for sub_file in os.listdir(folder_path):
        if sub_file.startswith(prefix) and os.path.isdir(
                os.path.join(folder_path, sub_file)) and sub_file != f'{prefix}_aggregated':
            for sub_sub_file in os.listdir(os.path.join(folder_path, sub_file)):
                # print(sub_sub_file)
                if sub_sub_file.endswith('.pickle'):
                    full_path = os.path.join(folder_path, sub_file, sub_sub_file)
                    shutil.copy(full_path, os.path.join(output_folder, sub_sub_file))
            if delete_segmented:
                shutil.rmtree(os.path.join(folder_path, sub_file))


def merge_EFOX_data(folder_path, all_formula_data_file):
    all_formula_data = pd.read_csv(all_formula_data_file)
    marinal_log, joint_log = fixed_depth_nested_dict(float, 2), fixed_depth_nested_dict(float, 2)
    EFO1_log = fixed_depth_nested_dict(float, 2)
    marginal_metric_list = ['marginal_MRR', 'marginal_HITS1', 'marginal_HITS3', 'marginal_HITS10', 'num_queries']
    joint_metric_list = ['couple_MRR', 'HITS1*1', 'HITS3*3', 'HITS10*10', 'MRR', 'HITS1', 'HITS3', 'HITS10',
                         'num_queries']
    EFO1_metric_list = ['MRR', 'HITS1', 'HITS3', 'HITS10', 'num_queries']
    for i, row in all_formula_data.iterrows():
        formula_id, formula = row['formula_id'], row['formula']
        with open(os.path.join(folder_path, f'all_logging_test_0_{formula_id}.pickle'), 'rb') as f:
            single_log = pickle.load(f)
            two_mar_log, one_mar_log, no_mar_log = single_log[formula]
            if row['f_num'] != 1:
                for key in marginal_metric_list:
                    marinal_log[formula][key] += two_mar_log[key]
                    marinal_log[formula][key] += one_mar_log[key]
                if marinal_log[formula]['num_queries'] != 0:
                    for key in marinal_log[formula]:
                        if key != 'num_queries':
                            marinal_log[formula][key] /= marinal_log[formula]['num_queries']
                for key in joint_metric_list:
                    joint_log[formula][key] += two_mar_log[key]
                    joint_log[formula][key] += one_mar_log[key]
                    joint_log[formula][key] += no_mar_log[key]
                for key in joint_log[formula]:
                    if key != 'num_queries':
                        joint_log[formula][key] /= joint_log[formula]['num_queries']
            else:
                EFO1_log[formula] = two_mar_log
                for key in EFO1_log[formula]:
                    if key != 'num_queries':
                        EFO1_log[formula][key] /= EFO1_log[formula]['num_queries']

    pd.DataFrame(marinal_log).T.to_csv(os.path.join(folder_path, 'marginal_log.csv'))
    pd.DataFrame(joint_log).T.to_csv(os.path.join(folder_path, 'joint_log.csv'))
    pd.DataFrame(EFO1_log).T.to_csv(os.path.join(folder_path, 'EFO1_log.csv'))
    return marinal_log, joint_log, EFO1_log

# output_dict = process_output_whole_folder('EFO-1_log/operator_MAML_LogicE/10_27', True, False)
# process_output_whole_folder('EFO-1_log/original4compare', False, False)


# process_output_whole_folder('EFO-1_log/original4compare', False, False)
# output_dict = process_output_whole_folder('EFO-1_log/operator_MAML_LogicE/10_10', False, False)
#
# aggregate_test('EFO-1_log/test_urgent', 'LogicE_0001_eval_False_lr_0.004', True)
# remove_checkpoint('EFO-1_log/operator_MAML_LogicE/11_9', [], True)
# remove_checkpoint('EFO-1_log/operator_MAML_LogicE/11_6', [], True)
fb237_result = {
    'valid_faithful': 'results/sparse/torch_0.01_0.01.ckpt230118.14:34:56ed0b73c5'
}

# process_output_whole_folder('results/sparse', ["step", "formula", "metric"], True, 'test', {'step': 0}, True, False)
merge_EFOX_data('EFO-1_log/LogicE_FB15k-237_EFOX.yaml230531.15:26:35192a92af', 'data/DNF_EFO2_23_4123166.csv')
