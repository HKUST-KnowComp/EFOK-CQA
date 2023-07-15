import copy
import json
from random import shuffle
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from src.language.foq import ConjunctiveFormula, DisjunctiveFormula, Disjunction, EFO1Query
from src.language.grammar import parse_lstr_to_lformula, parse_lstr_to_lformula_v2, concate_iu_chains, \
    parse_lstr_to_disjunctive_formula


class QAACollatorWithNoisySentencePair:
    def __init__(self, lstr, answer_size=-1, noisy_sample_size=-1):
        self.lstr = lstr
        self.answer_size = answer_size
        self.noisy_sample_size = noisy_sample_size

    def __call__(self, batch_input):
        lformula = parse_lstr_to_lformula(self.lstr)
        positive_fof = ConjunctiveFormula(lformula)
        lformula = parse_lstr_to_lformula(self.lstr)
        negative_fof = ConjunctiveFormula(lformula)

        for rsdict, easy_ans, _ in batch_input:
            positive_fof.append_qa_instances_as_sentence(rsdict,
                                                         answers=easy_ans)

            noisy_ans = {}
            for k in easy_ans:
                noisy_samples_tensor = torch.randint(
                    low=0, high=self.answer_size, size=(self.noisy_sample_size,))
                noisy_samples = noisy_samples_tensor.tolist()
                noisy_ans[k] = noisy_samples

            negative_fof.append_qa_instances_as_sentence(rsdict,
                                                         answers=noisy_ans)

        return positive_fof, negative_fof


class QAACollatorWithNoisyAnswers:
    def __init__(self, lstr, answer_size=-1, noisy_sample_size=-1):
        self.lstr = lstr
        self.answer_size = answer_size
        self.noisy_sample_size = noisy_sample_size

    def __call__(self, batch_input):
        lformula = parse_lstr_to_lformula(self.lstr)
        positive_fof = ConjunctiveFormula(lformula)
        lformula = parse_lstr_to_lformula(self.lstr)
        negative_fof = ConjunctiveFormula(lformula)

        for rsdict, easy_ans, _ in batch_input:
            positive_fof.append_qa_instances(rsdict,
                                             easy_answers=easy_ans)

            noisy_ans = {}
            for k in easy_ans:
                noisy_samples_tensor = torch.randint(
                    low=0, high=self.answer_size, size=(self.noisy_sample_size,))
                noisy_samples = noisy_samples_tensor.tolist()
                noisy_ans[k] = noisy_samples

            negative_fof.append_qa_instances(rsdict,
                                             easy_answers=noisy_ans)

        return positive_fof, negative_fof


class QAACollator:
    def __init__(self, lstr):
        self.lstr = lstr

    def __call__(self, batch_input):
        lformula = parse_lstr_to_lformula_v2(self.lstr)
        fof = EFO1Query(lformula)
        for rsdict, easy_ans, hard_ans in batch_input:
            fof.append_qa_instances(rsdict, easy_ans, hard_ans)
        return fof


class QAACollator_v2:
    def __init__(self, lstr):
        self.lstr = lstr

    def __call__(self, batch_input):
        fof = parse_lstr_to_disjunctive_formula(self.lstr)
        for rsdict, easy_ans, hard_ans in batch_input:
            fof.append_qa_instances(rsdict, easy_ans, hard_ans)
        return fof


class QueryAnsweringSeqDataLoader:
    def __init__(self, qaafile, target_lstr=None, size_limit=-1, **dataloader_kwargs) -> None:
        # FIXME: size_limit=-1 neglect the last one
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        for lstr, qaa in self.lstr_qaa.items():
            if target_lstr:
                if lstr not in target_lstr:
                    continue
            if not qaa:
                print(lstr, "query type is empty, continue")
                continue
            self.lstr_iterator[lstr] = DataLoader(qaa[:size_limit],
                                                  collate_fn=QAACollator(lstr),
                                                  **self.dataloader_kwargs)

    def get_fof_list(self):
        batch_buffer = []
        for _, iterator in self.lstr_iterator.items():
            for batch in iterator:
                batch_buffer.append(batch)
        shuffle(batch_buffer)
        return batch_buffer


class QueryAnsweringSeqDataLoader_v2:
    def __init__(self, qaafile, target_lstr=None, size_limit=None, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        for lstr, qaa in self.lstr_qaa.items():
            if target_lstr:
                if lstr not in target_lstr:
                    continue
            if not qaa:
                print(lstr, "query type is empty, continue")
                continue
            if size_limit:
                self.lstr_iterator[lstr] = DataLoader(qaa[:size_limit], collate_fn=QAACollator_v2(lstr),
                                                      **self.dataloader_kwargs)
            else:
                self.lstr_iterator[lstr] = DataLoader(qaa, collate_fn=QAACollator_v2(lstr),
                                                      **self.dataloader_kwargs)

    def get_fof_list(self):
        batch_buffer = []
        for _, iterator in self.lstr_iterator.items():
            for batch in iterator:
                batch_buffer.append(batch)
        shuffle(batch_buffer)
        return batch_buffer

    def get_fof_list_no_shuffle(self):
        batch_buffer = []
        for _, iterator in self.lstr_iterator.items():
            for batch in iterator:
                batch_buffer.append(batch)
        return batch_buffer


class QueryAnsweringMixIterator:
    def __init__(self, qaafile, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)
        self.lstr_iterator = {}
        samples_per_query = {}
        total_samplers = 0
        for k in self.lstr_qaa:
            size_k = len(self.lstr_qaa[k])
            samples_per_query[k] = size_k
            total_samplers += size_k

        total_num_iterations = total_samplers // dataloader_kwargs.pop('batch_size') + 1

        self.batch_size_per_query = {
            k: samples_per_query[k] // total_num_iterations + 1
            for k in samples_per_query}

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            if not qaa: continue
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                                                       batch_size=self.batch_size_per_query[lstr],
                                                       collate_fn=QAACollator(lstr),
                                                       **self.dataloader_kwargs))

        return self

    def __next__(self):
        buffer = []
        for _, dataloader in self.lstr_iterator.items():
            try:
                buffer.append(next(dataloader))
            except StopIteration:
                pass

        if len(buffer) == 0:
            raise StopIteration

        return buffer

    def __len__(self):
        return sum([len(iterator) for iterator in self.lstr_iterator.values()])


class QueryAnsweringMixDataLoader:
    def __init__(self, qaafile, target_lstr=None, size_limit = None, **dataloader_kwargs) -> None:
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)
        self.lstr_iterator = OrderedDict()
        self.lstr_data = OrderedDict()
        samples_per_query = {}
        total_samplers = 0
        for k in self.lstr_qaa:
            if target_lstr:
                if k not in target_lstr:
                    continue
            size_k = len(self.lstr_qaa[k])
            samples_per_query[k] = size_k
            total_samplers += size_k

        total_num_iterations = total_samplers // dataloader_kwargs['batch_size'] + 1

        self.batch_size_per_query = {
            k: samples_per_query[k] // total_num_iterations + 1
            for k in samples_per_query}
        for lstr, qaa in self.lstr_qaa.items():
            if target_lstr:
                if lstr not in target_lstr:
                    continue
            if not qaa:
                print(lstr, "query type is empty, continue")
                continue
            lstr_kwargs = copy.deepcopy(self.dataloader_kwargs)
            lstr_kwargs['batch_size'] = self.batch_size_per_query[lstr]
            if size_limit:
                self.lstr_data[lstr] = DataLoader(qaa[:size_limit], collate_fn=QAACollator_v2(lstr),
                                                  **lstr_kwargs)
                self.lstr_iterator[lstr] = iter(DataLoader(qaa[:size_limit], collate_fn=QAACollator_v2(lstr),
                                                           **lstr_kwargs))
            else:
                self.lstr_data[lstr] = DataLoader(qaa, collate_fn=QAACollator_v2(lstr), **lstr_kwargs)
                self.lstr_iterator[lstr] = iter(DataLoader(qaa, collate_fn=QAACollator_v2(lstr), **lstr_kwargs))

    def get_single_fof_list_no_shuffle(self):
        batch_buffer = {}
        for lstr in self.lstr_iterator:
            iterator = self.lstr_iterator[lstr]
            batch = next(iter(iterator))
            batch_buffer[lstr] = batch
        return batch_buffer

    def get_single_fof_list(self):
        batch_buffer = {}
        for lstr in self.lstr_iterator:
            iterator = self.lstr_iterator[lstr]
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.lstr_data[lstr])
                self.lstr_iterator[lstr] = iterator
                batch = next(iterator)
            batch_buffer[lstr] = batch
        return batch_buffer

    def __len__(self):
        return sum([len(iterator) for iterator in self.lstr_data.values()])

# fixme: use when needed
class TrainRandomSentencePairDataLoader:
    def __init__(self,
                 qaafile,
                 answer_size,
                 noisy_sample_size,
                 **dataloader_kwargs) -> None:
        self.qaafile = qaafile
        self.answer_size = answer_size
        self.noisy_sample_size = noisy_sample_size
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        self.batch_buffer = []

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            if not qaa: continue
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                                                       collate_fn=QAACollatorWithNoisySentencePair(
                                                           lstr, self.answer_size, self.noisy_sample_size),
                                                       **self.dataloader_kwargs))
        return self

    def __next__(self):
        if len(self.batch_buffer) == 0:
            for lstr, iterator in self.lstr_iterator.items():
                try:
                    self.batch_buffer.append(next(iterator))
                except StopIteration:
                    pass

            if len(self.batch_buffer) == 0:
                raise StopIteration
            else:
                shuffle(self.batch_buffer)

        return self.batch_buffer.pop()

    def __len__(self):
        return sum([len(iterator) for iterator in self.lstr_iterator.values()])


class TrainNoisyAnswerDataLoader:
    def __init__(self,
                 qaafile,
                 answer_size,
                 noisy_sample_size,
                 **dataloader_kwargs) -> None:
        self.qaafile = qaafile
        self.answer_size = answer_size
        self.noisy_sample_size = noisy_sample_size
        self.dataloader_kwargs = dataloader_kwargs

        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_iterator = {}
        self.batch_buffer = []

    def __iter__(self):
        for lstr, qaa in self.lstr_qaa.items():
            if not qaa: continue
            self.lstr_iterator[lstr] = iter(DataLoader(qaa,
                                                       collate_fn=QAACollatorWithNoisyAnswers(
                                                           lstr, self.answer_size, self.noisy_sample_size),
                                                       **self.dataloader_kwargs))
        return self

    def __next__(self):
        if len(self.batch_buffer) == 0:
            for lstr, iterator in self.lstr_iterator.items():
                try:
                    self.batch_buffer.append(next(iterator))
                except StopIteration:
                    pass

            if len(self.batch_buffer) == 0:
                raise StopIteration
            else:
                shuffle(self.batch_buffer)

        return self.batch_buffer.pop()

    def __len__(self):
        return sum([len(iterator) for iterator in self.lstr_iterator.values()])


