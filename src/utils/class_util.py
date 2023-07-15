import collections
import hashlib
from functools import partial
from itertools import repeat
import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import join, dirname, realpath, exists
from shutil import rmtree
import time
import pickle
import json
import numpy as np
import pandas as pd

dir_path = dirname(realpath(__file__))


def nested_dict(): return collections.defaultdict(nested_dict)


def fixed_depth_nested_dict(default_factory, depth=1):
    result = partial(collections.defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(collections.defaultdict, result)
    return result()


def rename_ordered_dict(old_dict, old_key, new_key):
    """
    Create a new OrderedDict for rename a given key in old OrderedDict
    """
    new_dict = collections.OrderedDict((new_key if k == old_key else k, v) for k, v in old_dict.items())
    return new_dict


def compare_torch_dict(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    final_compare_dict = {}
    for key in dict1:
        if isinstance(dict1[key], dict):
            sub_compare = compare_torch_dict(dict1[key], dict2[key])
            final_compare_dict[key] = sub_compare
        elif isinstance(dict1[key], torch.Tensor):
            final_compare_dict[key] = torch.all(dict1[key] == dict2[key])
        else:
            final_compare_dict[key] = (dict1[key] == dict2[key])
    return final_compare_dict


class Writer:
    _log_path = join(dir_path, 'log')

    def __init__(self, case_name, config, log_path=None, postfix=True, tb_writer=None):
        if isinstance(config, dict):
            self.meta = config
        else:
            self.meta = vars(config)
        self.time = time.time()
        self.meta['time'] = self.time
        self.idstr = case_name
        self.column_name = {}
        if postfix:
            self.idstr += time.strftime("%y%m%d.%H:%M:%S", time.localtime()) + \
                          hashlib.sha1(str(self.meta).encode('UTF-8')).hexdigest()[:8]

        self.log_path = log_path if log_path else self._log_path
        if exists(self.case_dir):
            rmtree(self.case_dir)
        os.makedirs(self.case_dir, exist_ok=False)

        with open(self.metaf, 'wt') as f:
            json.dump(self.meta, f)

    def append_trace(self, trace_name, data):
        if trace_name not in self.column_name:
            self.column_name[trace_name] = list(data.keys())
            assert len(self.column_name[trace_name]) > 0
        if not os.path.exists(self.tracef(trace_name)):
            with open(self.tracef(trace_name), 'at') as f:
                f.write(','.join(self.column_name[trace_name]) + '\n')
        with open(self.tracef(trace_name), 'at') as f:
            f.write(','.join([str(data[c]) for c in self.column_name[trace_name]]) + '\n')

    def save_pickle(self, obj, name):
        with open(join(self.case_dir, name), 'wb') as f:
            pickle.dump(obj, f)

    def save_array(self, arr, name):
        np.save(join(self.case_dir, name), arr)

    def save_json(self, obj, name):
        if not name.endswith('json'):
            name += '.json'
        with open(join(self.case_dir, name), 'wt') as f:
            json.dump(obj, f)

    def save_dataframe(self, obj, name):
        if not name.endswith('csv'):
            name += '.csv'
        df = pd.DataFrame.from_dict(data=obj)
        df.to_csv(join(self.case_dir, name))

    def save_torch(self, obj, name):
        if not name.endswith('ckpt'):
            name += '.ckpt'
        torch.save(obj, join(self.case_dir, name))

    def save_model(self, model: torch.nn.Module, opt, step, warm_up_step, lr):
        print("saving model : ", step)
        device = model.device
        save_data = {'model_parameter': model.cpu().state_dict(), 'optimizer_parameter': opt.state_dict(),
                     'warm_up_steps': warm_up_step, 'learning_rate': lr}
        torch.save(save_data, self.modelf(step))
        model.to(device)

    def save_plot(self, fig, name):
        fig.savefig(join(self.case_dir, name))

    @property
    def case_dir(self):
        return join(self.log_path, self.idstr)

    @property
    def metaf(self):
        return join(self.case_dir, 'meta.json')

    def tracef(self, name):
        return join(self.case_dir, '{}.csv'.format(name))

    def modelf(self, e):
        return join(self.case_dir, '{}.ckpt'.format(e))
