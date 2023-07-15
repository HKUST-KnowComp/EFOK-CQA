import json
import os
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class JSONlRecorder:
    def __init__(self, jsonl_filename):
        self.jsonl_filename = jsonl_filename

    def append_record(self, data: Dict):
        line_content = json.dumps(data)
        with open(self.jsonl_filename, mode='at') as f:
            f.write(f"{line_content}\n")


class Recorder:
    def __init__(self, logdir, prefix):
        self._tb_writer = SummaryWriter(logdir)
        self._jsonl_recorder = JSONlRecorder(os.path.join(logdir, prefix + '.jsonl'))

    def tb_write(self, data: Dict, global_step: int):
        for k, v in data.items():
            self._tb_writer.add_scalar(tag=k, scalar_value=v,
                                       global_step=global_step)

    def jsonl_write(self, data: Dict):
        self._jsonl_recorder.append_record(data)

    def write(self, data):
        self.jsonl_write(data)
        if 'global_step' in data:
            self.tb_write(data, global_step=data['global_step'])
        elif 'step' in data:
            self.tb_write(data, global_step=data['step'])


class TrainRecorder(Recorder):
    def __init__(self, logdir):
        super().__init__(logdir, "train")


class EvalRecorder(Recorder):
    def __init__(self, logdir, task_name):
        super().__init__(logdir, task_name)
