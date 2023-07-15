import logging
import os

import torch

from .evaluation import Evaluator
from .structure import KnowledgeGraph, NeuralBinaryPredicate
from .learner import Learner, LearnerForwardOutput
from .utils.recorder import TrainRecorder
from .utils.config import ExperimentConfigCollection


class Trainer:
    """
    Basic interface for training and evaluate the model
    basic objects
        - kg
        - nbp
        - learner
        - optimizer
        - evaluator
        - recorder
    """

    def __init__(self,
                 logdir: str,
                 kg: KnowledgeGraph,
                 nbp: NeuralBinaryPredicate,
                 learner: Learner,
                 optimizer: torch.optim.Optimizer,
                 dev_evaluator: Evaluator,
                 test_evaluator: Evaluator,
                 recorder: TrainRecorder,
                 objective='nce',
                 margin=10,
                 k_nce=1,
                 num_neg_samples=1,
                 ns_strategy='lcwa',
                 batch_size=256,
                 num_steps=10000,
                 num_epochs=1000,
                 **kwargs):
        # important objects
        self.logdir = logdir
        self.kg = kg
        self.nbp = nbp
        self.learner = learner
        self.optimizer = optimizer
        self.dev_evaluator = dev_evaluator
        self.test_evaluator = test_evaluator
        self.recorder = recorder
        # parameters
        self.objective = objective
        self.margin = margin
        self.num_neg_samples = num_neg_samples
        self.k_nce = k_nce
        self.ns_strategy = ns_strategy
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        # internal fields
        self._iterator = None
        self.epoch = -1
        self._epoch_eval_flag = 0
        self.step = 0

    @classmethod
    def create(cls, ecc: ExperimentConfigCollection):
        ecc.show_config()

        # create the KnowledgeGraph
        logging.info(f"create the (observed) knowledge graph")
        logging.info(f"\t {ecc.knowledge_graph_config.to_dict()}")
        kg = KnowledgeGraph.from_config(ecc.knowledge_graph_config)
        logging.info(f"kg created")

        # create the neural binary predicate
        logging.info(f"create the neural binary predicate")
        logging.info(f"\t {ecc.neural_binary_predicate_config.to_dict()}")
        nbp = ecc.neural_binary_predicate_config.instantiate(kg)
        logging.info(f"nbp created")

        # create learner
        logging.info(f"create the learner")
        logging.info(f"\t {ecc.learner_config.to_dict()}")
        learner = ecc.learner_config.instantiate(kg, nbp)
        logging.info(f"learner created")

        # create the optimizer
        logging.info(f"create the optimizer")
        logging.info(f"\t {ecc.optimizer_config.to_dict()}")
        optimizer = ecc.optimizer_config.instantiate(nbp.parameters())
        logging.info(f"optimizer created")

        # create the evaluator
        dev_evaluator = Evaluator.create(ecc.dev_evaluation_config, ecc.logdir)
        test_evaluator = Evaluator.create(ecc.test_evaluation_config, ecc.logdir)

        # create the train recorder
        recorder = TrainRecorder(ecc.logdir)

        # create trainer
        trainer = cls(logdir=ecc.logdir,
                      kg=kg,
                      nbp=nbp,
                      learner=learner,
                      optimizer=optimizer,
                      dev_evaluator=dev_evaluator,
                      test_evaluator=test_evaluator,
                      recorder=recorder,
                      **ecc.trainer_config.to_dict())

        return trainer

    def get_next_batch_input(self):
        try:
            if self._iterator is None: raise StopIteration
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._epoch_eval_flag = 0
            print("train epoch", self.epoch)
            self._iterator = self.learner.get_data_iterator(
                batch_size=self.batch_size,
                shuffle=True)
            batch = next(self._iterator)
        return batch

    def _compute_nce_loss(self, batch_output: LearnerForwardOutput):
        loss = 0
        loss -= torch.log(batch_output.pos_prob.mean(-1))
        loss -= torch.log(1 - batch_output.neg_prob.mean(-1))
        return loss.mean()

    def _compute_pairwise_loss(self, batch_output: LearnerForwardOutput):
        loss = self.margin
        loss += batch_output.neg_score.mean(-1)
        loss -= batch_output.pos_score.mean(-1)
        loss = torch.relu(loss).mean()
        return loss

    def train_step(self):
        log = {}

        self.optimizer.zero_grad()

        batch_input = self.get_next_batch_input()

        batch_output = self.learner.forward(
            batch_input, self.num_neg_samples, self.ns_strategy, self.margin)

        if self.objective == 'nce':
            loss = self._compute_nce_loss(batch_output)

        elif self.objective == 'pairwise':
            loss = self._compute_pairwise_loss(batch_output)

        else:
            raise NotImplementedError(
                f"Unknown loss function {self.objective}")

        loss.backward()
        self.optimizer.step()
        self.step += 1

        log['loss'] = loss.item()
        log['epoch'] = self.epoch
        log['step'] = self.step

        return log

    def _not_finish_train(self):
        if self.num_steps > 0:
            return self.step < self.num_steps
        elif self.num_epochs > 0:
            return self.epoch < self.num_epochs
        else:
            raise NotImplementedError

    def _should_eval(self):
        if self.dev_evaluator.eval_every_step > 0:
            return (self.step + 1) % self.dev_evaluator.eval_every_step == 0
        if self.dev_evaluator.eval_every_epoch > 0 and self._epoch_eval_flag == 0:
            self._epoch_eval_flag = 1
            return (self.epoch + 1) % self.dev_evaluator.eval_every_epoch == 0
        return False

    def run(self):
        best_key_metric = self.dev_evaluator.evaluate_nbp(
            self.nbp, self.step, self.epoch, get_key_metric=True)
        while self._not_finish_train():
            log = self.train_step()
            self.recorder.write(log)
            if self._should_eval():
                key_metric = self.dev_evaluator.evaluate_nbp(
                    self.nbp, self.step, self.epoch, get_key_metric=True)
                self.test_evaluator.evaluate_nbp(
                    self.nbp, self.step, self.epoch, get_key_metric=False)

                if key_metric > best_key_metric:
                    new_path = os.path.join(self.logdir, f'step={self.step}:epoch={self.epoch}.ckpt')
                    torch.save(self.nbp.state_dict(), new_path)
                    logging.info(f"key metric ({self.dev_evaluator.dev_key}) = {key_metric} is better than {best_key_metric}, new checkpoint saved to {new_path}")
                    best_key_metric = key_metric
